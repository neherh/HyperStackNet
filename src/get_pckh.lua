--1. get pckh values 
--2. comment line 77 and uncomment line 76 of eval.lua when running this to get joint -wise accuracy
--3. revert point 2 above when running model 

require 'paths'
require 'hdf5'


projectDir = paths.concat(os.getenv('HOME'),'pose-hg-train')
paths.dofile('ref.lua')
paths.dofile('util/eval.lua')
paths.dofile('util/pose.lua')
paths.dofile('util/dataloader.lua')

idxs, preds, hms, inp = loadPreds('harpe/test-runCorrect182/final_preds', true, true)

--print(preds:numel())
--print(inp:size())
--print(hms:size())
--print( type(#idxs))

--preds1 = getPreds(hms)

--print(preds1:size())


inp, label = loadData(nil , idxs)

print("Label dimensions")
print(label[8]:size())
label = label[8]

--local gt = getPreds(label)


--basic_acc = basicAccuracy(hms, label, nil)
--print("Basic Accuracy:")
--print(basic_acc)

--print("Dist")


--dists = calcDists(preds1, gt, torch.ones(preds1:size(1))*64/10)
--print(dists:size())
--f = distAccuracy(dists, nil)

f = heatmapAccuracy(hms , label , nil , dataset.accIdxs)

print("******")
print(f)

