--Code for generating vizualization of labels 


require 'paths'
require 'hdf5'


projectDir = paths.concat(os.getenv('HOME'),'pose-hg-train')
paths.dofile('ref.lua')

paths.dofile('util/eval.lua')
paths.dofile('util/pose.lua')
paths.dofile('util/dataloader.lua')
paths.dofile('util/img.lua')
 
function drawOutput(input, hms)
    local im = input:clone()
    local colorHms = {}
    local inp64 = image.scale(input,64):mul(.3)
    for i = 1,18 do 
        colorHms[i] = colorHM(hms[i])
        colorHms[i]:mul(.7):add(inp64)
    end
    local totalHm = compileImages18(colorHms, 5, 4, 64)
    --im = compileImages({im,totalHm}, 1, 2, 256)
    totalHm = image.scale(totalHm,756)
    return totalHm
end

idxs = torch.range(1, 886)
local a = hdf5.open('../data/harpe/annot_corrected.h5')
annot = {}
local tags = {'imgname','center','scale'}
for _,tag in ipairs(tags) do annot[tag] = a:read(tag):all() end
local nsamples = idxs:nElement()

__, labels = loadData(nil , idxs)
labels = labels[#labels]

--print(labels[#labels]:size())


for i =1,nsamples do

	local im = image.load('../data/harpe/images/' .. ffi.string(annot['imgname'][idxs[i]]:char():data()))
	local center = annot['center'][idxs[i]]
    local scale = annot['scale'][idxs[i]]
    scale = scale * 1.8
    local inp = crop(im, center, scale, 0, 256)
    labels[i]:mul(4)

    local dispImg = drawOutput(inp, labels[i])

     image.save('labels/' .. ffi.string(annot['imgname'][idxs[i]]:char():data()) , dispImg)
     xlua.progress(i,nsamples)
     --sys.sleep(3)
end












