require 'paths'
require 'hdf5'


projectDir = paths.concat(os.getenv('HOME'),'pose-hg-train')
paths.dofile('ref.lua')
paths.dofile('util/eval.lua')
paths.dofile('util/pose.lua')
paths.dofile('util/dataloader.lua')


local f = hdf5.open('harpe_pred.h5','r')
local hms = f:read('preds'):all()

idxs, preds, hms_, inp = loadPreds('harpe/test-runNewAnnot1/final_preds', true, true)

idxs = torch.sort(idxs)
--idxs = idxs[idxs:gt(182)]

inp, label = loadData(nil , idxs)

label = label[8]
pckh = heatmapAccuracy(hms , label , nil , dataset.accIdxs)

print("******")
print(label:size())
print(pckh)