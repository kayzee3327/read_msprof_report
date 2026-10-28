import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

from ascend_analyzers.ub import analyze_document, union_bytes, write_outputs


def row(offset=0, size=8, start=1, end=3, tmp=False, name="%a"):
    return dict(buffer=f"{name} = memref.alloc() : memref<{size}xi8>",
                extent=size * 8, offset=[offset], alloc_time_in_ir=start,
                life_time_in_ir=[start, end], is_tmpbuf=tmp, source_location="loc(test)")


def document(rows):
    return dict(Header=dict(KernelName="test"), Record=[dict(scope="ub", status="success", error_info="", memory_info_array=rows)])


class AnalysisTests(unittest.TestCase):
    def analyze(self, rows):
        return analyze_document(document(rows))["records"][0]

    def test_sample_regression(self):
        doc = json.loads(Path(__file__).with_name("memory_info_aiv.json").read_text())
        r = analyze_document(doc)["records"][0]
        self.assertEqual((r["raw_rows"], r["duplicate_rows_removed"], r["storage_groups"], r["physical_instances"]), (28, 6, 18, 23))
        self.assertEqual(r["address_high_water_bytes"], 77632)
        self.assertEqual(r["peak_live"], dict(bytes=39040, ir_ranges=[[95, 95]]))
        self.assertEqual(r["peak_tmp"]["bytes"], 4096)
        self.assertEqual(r["overlaps"], [])

    def test_closed_lifetime_and_single_point_tmp(self):
        r = self.analyze([row(0, 8, 1, 3), row(8, 4, 3, 3, True), row(12, 8, 3, 5)])
        self.assertEqual(r["peak_live"], dict(bytes=20, ir_ranges=[[3, 3]]))
        self.assertEqual(r["timeline"][-1]["live_bytes"], 0)

    def test_multibuffer_duplicates_and_aliases(self):
        a = row(); a["offset"] = [0, 100]
        b = copy.deepcopy(a); b["buffer"] = b["buffer"].replace("%a", "%alias")
        r = self.analyze([a, copy.deepcopy(a), b])
        self.assertEqual(r["peak_live"]["bytes"], 16)
        self.assertEqual(r["address_high_water_bytes"], 108)
        self.assertEqual(r["storage_groups"], 1)
        self.assertEqual(len(r["groups"][0]["aliases"]), 2)

    def test_partial_overlap_and_tmp_attribution(self):
        r = self.analyze([row(0, 10), row(5, 10, tmp=True)])
        self.assertEqual(r["peak_live"]["bytes"], 15)
        self.assertEqual(r["peak_tmp"]["bytes"], 10)
        self.assertEqual(r["peak_tmp_incremental"]["bytes"], 5)
        self.assertEqual(len(r["overlaps"]), 1)

    def test_same_name_different_storage_and_reuse(self):
        r = self.analyze([row(0, 8, 1, 2), row(0, 8, 3, 4), row(20, 4, 3, 4)])
        self.assertEqual(r["storage_groups"], 3)
        self.assertEqual(r["peak_live"]["bytes"], 12)
        self.assertEqual(r["overlaps"], [])

    def test_extent_overrides_memref_shape(self):
        a = row(); a["extent"] = 128
        r = self.analyze([a])
        self.assertEqual(r["peak_live"]["bytes"], 16)
        self.assertTrue(r["warnings"])

    def test_records_are_independent(self):
        d = document([row()]); d["Record"].append(document([row(100)])["Record"][0])
        self.assertEqual([r["peak_live"]["bytes"] for r in analyze_document(d)["records"]], [8, 8])
        d["Record"].append(dict(scope="gm", status="fail"))
        self.assertEqual(len(analyze_document(d)["records"]), 2)

    def test_reject_malformed_and_failed_plans(self):
        changes = [dict(offset=[]), dict(offset=[-1]), dict(extent=7), dict(extent=True),
                   dict(life_time_in_ir=[4, 1]), dict(life_time_in_ir=[1]), dict(is_tmpbuf="false")]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                a = row(); a.update(change); self.analyze([a])
        d = document([row()]); d["Record"][0]["status"] = "fail"
        with self.assertRaises(ValueError):
            analyze_document(d)
        with self.assertRaises(ValueError):
            analyze_document(document([]), -1)

    def test_empty_and_zero_size(self):
        self.assertEqual(self.analyze([])["peak_live"]["bytes"], 0)
        self.assertEqual(self.analyze([row(100, 0)])["address_high_water_bytes"], 0)

    def test_sweep_matches_brute_force(self):
        rng = random.Random(42)
        for _ in range(40):
            rows = [row(rng.randrange(30), rng.randrange(1, 10), s := rng.randrange(10), rng.randrange(s, 12), bool(rng.randrange(2)), f"%{i}") for i in range(15)]
            r = self.analyze(rows)
            for s in r["timeline"]:
                for t in range(s["start"], s["end"] + 1):
                    occupied, normal, tmp = set(), set(), set()
                    for a in rows:
                        if a["life_time_in_ir"][0] <= t <= a["life_time_in_ir"][1]:
                            addresses = set(range(a["offset"][0], a["offset"][0] + a["extent"] // 8))
                            occupied |= addresses
                            (tmp if a["is_tmpbuf"] else normal).update(addresses)
                    self.assertEqual(s["live_bytes"], len(occupied))
                    self.assertEqual(s["tmp_union_bytes"], len(tmp))
                    self.assertEqual(s["tmp_incremental_bytes"], len(occupied - normal))

    def test_output_is_offline_and_script_safe(self):
        d = document([row()]); d["Header"]["KernelName"] = "</script><script>alert(1)</script>"
        with tempfile.TemporaryDirectory() as directory:
            write_outputs(analyze_document(d), directory)
            output = Path(directory, "report.html").read_text()
            self.assertNotIn(d["Header"]["KernelName"], output)
            self.assertNotIn('<script src=', output)
            self.assertEqual(json.loads(Path(directory, "analysis.json").read_text())["kernel"], d["Header"]["KernelName"])
            self.assertTrue(Path(directory, "timeline.csv").read_text().startswith("record,ir_start"))

    def test_interval_union(self):
        self.assertEqual(union_bytes([(0, 5), (2, 3), (5, 8), (20, 22)]), 10)


if __name__ == "__main__":
    unittest.main()
