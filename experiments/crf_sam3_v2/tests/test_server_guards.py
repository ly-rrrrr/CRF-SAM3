"""Regression tests for the server-specific failure modes (CPU only)."""
import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(HERE),str(HERE/'tools')]
from make_verify_config import scan
from ground_infer_plan import main as merge
from v2.validation import validate_heat,validate_cache

class Guards(unittest.TestCase):
 def test_nested_revos_scan(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);p=root/'OVIS/train/abc/12/meta.json';p.parent.mkdir(parents=True)
   p.write_text(json.dumps({'video_id':'OVIS/train/abc','obj_ids':[0,1,1000]}))
   self.assertEqual(scan(root),(1,1,3))
 def test_nan_heat_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'x.npz';np.savez(p,responses=[np.nan],image_grid_thw=[[1,2,2]],frame_stems=['0'],merge=2)
   with self.assertRaises(ValueError):validate_heat(p)
 def test_merge_requires_heat(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);p=root/'revos/full';p.mkdir(parents=True)
   (p/'meta_p0.json').write_text(json.dumps({'videos':{'OVIS/train/a':{'expressions':{'0':{}}}}}))
   (p/'categories_p0.json').write_text(json.dumps({'records':[{'video_id':'OVIS/train/a','exp_id':'0','pred_category':'cat'}]}))
   self.assertEqual(merge(['merge','--out',t]),1)
   self.assertFalse((p/'categories.json').exists())
 def test_partial_cache_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);sp=root/'split.json';sp.write_text(json.dumps({'datasets':{'revos':{'train':['OVIS/train/a']}}}))
   with self.assertRaises(ValueError):validate_cache(root,sp,'revos','train',{'OVIS/train/a':{'expressions':{'0':{}}}})
 def test_exclusions_count_as_completed_not_training(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);sp=root/'split.json';sp.write_text(json.dumps({'datasets':{'revos':{'train':['OVIS/train/a']}}}))
   (root/'done_train_0of2.jsonl').write_text(json.dumps({'key':'OVIS/train/a/0','status':'no_positive'})+'\n')
   self.assertEqual(validate_cache(root,sp,'revos','train',{'OVIS/train/a':{'expressions':{'0':{}}}}),{'no_positive':1})
if __name__=='__main__':unittest.main()
