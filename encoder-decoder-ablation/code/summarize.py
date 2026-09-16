from runtime import *
import statistics,zipfile

def main():
 p,reg=verified();rows=read(ROOT/'evaluation_summary.json')['rows'];assert len(rows)==216 and len({(x['key'],x['id']) for x in rows})==216
 per={}
 for c in p['cases']:
  rr=[x for x in rows if x['key']==c['key']];ours=next(x for x in rr if x['id']=='ours');assert ours['video_sha256']==c['full_video_sha256']
  enc={};dec={}
  for pol in p['encoder_policies']:
   q=[x for x in rr if x['policy']==pol and x['panel'] in ['encoder','shared']];assert len(q)==(1 if pol in ['roi','ours'] else 3)
   enc[pol]={m:statistics.mean(x['metrics'][m] for x in q) for m in p['metrics']}
  for arm in p['decoder_arms']:
   q=[x for x in rr if x['policy']==arm and x['panel'] in ['decoder','shared']];assert len(q)==1;dec[arm]={m:q[0]['metrics'][m] for m in p['metrics']}
  assert enc['ours']==dec['ours'];per[c['key']]=dict(encoder=enc,decoder=dec,ours_video_sha256=ours['video_sha256'])
 tables={panel:{a:{m:statistics.mean(v[panel][a][m] for v in per.values()) for m in p['metrics']} for a in per[p['cases'][0]['key']][panel]} for panel in ['encoder','decoder']};assert tables['encoder']['ours']==tables['decoder']['ours']
 save(ROOT/'results.json',dict(tables=tables,per_sample=per,rows=rows,shared_ours=True,logical_conditions=120,raw_conditions=216))
 lines=['# Encoder / Decoder 统一消融','', '四个prompt，每个两个生成种子；encoder的六种随机/覆盖策略先对三个调度重复平均，再对八样本等权平均。ROI固定偏移404。两部分共享同一Ours输出及native参考。','']
 for panel in ['encoder','decoder']:
  lines += ['## '+panel,'','| 设置 | LPIPS ↓ | SSIM ↑ | PSNR ↑ |','|---|---:|---:|---:|']
  for a,t in tables[panel].items():lines.append(f"| {a} | {t['lpips']:.3f} | {t['ssim']:.3f} | {t['psnr_db']:.3f} |")
  lines.append('')
 lines += ['encoder共享逐步更新数量，decoder共享精确更新掩码。Decoder A是K/V插值开关，公式4潜变量对齐始终开启。所有样本和结果保留；此表是既有四场景上的实验，不声称统计显著性或跨场景普遍最优。']
 (ROOT/'report_cn.md').write_text('\n'.join(lines),encoding='utf-8')
 save(ROOT/'final_audit.json',dict(passed=True,raw_rows=216,logical_conditions=120,shared_ours_exact=True,samples=8))
 status('all_complete',raw_rows=216,logical_conditions=120)
 with zipfile.ZipFile(ROOT/'evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in ROOT.rglob('*'):
   if f.is_file() and f.suffix in ['.py','.json','.md','.log']:z.write(f,f.relative_to(ROOT))
if __name__=='__main__':main()
