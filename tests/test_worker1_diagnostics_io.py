import csv, importlib.util, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def load(name):
    p=ROOT/'saddlemill'/f'{name}.py'; s=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m); return m
io_mod=load('diagnostics_io')


def test_buffered_csv_preserves_rows_and_reduces_writes(tmp_path):
    path=tmp_path/'x.csv'; fields=['a','b']; w=io_mod.BufferedCSVAppender(path,fields,max_records=8)
    for i in range(25): w.append({'a':i,'b':i*i})
    w.close()
    with path.open(newline='') as h: rows=list(csv.DictReader(h))
    assert len(rows)==25 and rows[-1]['b']=='576'
    assert w.stats.write_calls <= 4
    assert w.stats.open_calls <= 1


def test_additive_csv_header_migration(tmp_path):
    path=tmp_path/'x.csv'; path.write_text('a\n1\n')
    w=io_mod.BufferedCSVAppender(path,['a','b'],max_records=4); w.append({'a':2,'b':3}); w.close()
    with path.open(newline='') as h: rows=list(csv.DictReader(h))
    assert rows == [{'a':'1','b':''},{'a':'2','b':'3'}]


def test_buffered_jsonl_is_bounded_and_equivalent(tmp_path):
    path=tmp_path/'x.jsonl'; w=io_mod.BufferedJSONLAppender(path,max_records=5)
    for i in range(13): w.append({'i':i})
    w.close(); rows=[json.loads(x) for x in path.read_text().splitlines()]
    assert [r['i'] for r in rows]==list(range(13)); assert w.stats.write_calls==3


def test_passive_instrumentation_does_not_change_synthetic_trajectory(tmp_path):
    def run(recorder=None):
        x=1.0; traj=[]
        for step in range(20):
            grad=2*x
            if abs(grad)<0.02: break
            x=x-0.2*grad; traj.append(x)
            if recorder: recorder({'step':step,'x':x,'grad':grad})
        return x,traj,step
    writer=io_mod.BufferedJSONLAppender(tmp_path/'diag.jsonl',max_records=4)
    a=run(); b=run(writer.append); writer.close()
    assert a==b

def test_csv_and_jsonl_bytes_match_predecessor_serialization(tmp_path):
    csv_path=tmp_path/'buffered.csv'; csv_expected=tmp_path/'expected.csv'
    fields=['a','b']
    rows=[{'a':1,'b':'x'},{'a':2,'b':'y'}]
    with csv_expected.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
        for row in rows: w.writerow(row)
    w=io_mod.BufferedCSVAppender(csv_path,fields,max_records=16)
    for row in rows: w.append(row)
    w.close()
    assert csv_path.read_bytes()==csv_expected.read_bytes()

    json_path=tmp_path/'buffered.jsonl'; json_expected=tmp_path/'expected.jsonl'
    expected=''.join(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n' for row in rows)
    json_expected.write_text(expected)
    j=io_mod.BufferedJSONLAppender(json_path,max_records=16)
    for row in rows: j.append(row)
    j.close()
    assert json_path.read_bytes()==json_expected.read_bytes()


def test_close_flushes_sub_batch_records(tmp_path):
    path=tmp_path/'small.jsonl'
    w=io_mod.BufferedJSONLAppender(path,max_records=16)
    for i in range(3): w.append({'i':i})
    assert not path.exists()
    w.close()
    assert [json.loads(x)['i'] for x in path.read_text().splitlines()]==[0,1,2]
