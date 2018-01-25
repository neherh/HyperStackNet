local M = {}
Dataset = torch.class('pose.Dataset',M)

function Dataset:__init()
    self.nJoints = 18
    self.accIdxs = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}
    self.flipRef = {{1,6},   {2,5},   {3,4},
                    {11,16}, {12,15}, {13,14}}
    -- Pairs of joints for drawing skeleton
    self.skeletonRef = {{1,2,1},    {2,3,1},    {3,7,1},
                        {4,5,2},    {4,7,2},    {5,6,2},
                        {7,9,0},    {9,10,0},
                        {13,9,3},   {11,12,3},  {12,13,3},
                        {11,17,1},  {17,18,1},
                        {14,9,4},   {14,15,4},  {15,16,4}}

    local annot = {}
    local tags = {'index','person','imgname','part','center','scale',
                  'normalize','torsoangle','visible','multi','istrain'}
    local a = hdf5.open(paths.concat(projectDir,'data/harpe/annot.h5'),'r')
    for _,tag in ipairs(tags) do annot[tag] = a:read(tag):all() end
    a:close()
    annot.index:add(1)
    annot.person:add(1)
    annot.part:add(1)

    -- Index reference
    if not opt.idxRef then
        local allIdxs = torch.range(1,annot.index:size(1))
        opt.idxRef = {}
        opt.idxRef.test = allIdxs[annot.istrain:eq(0)]
        opt.idxRef.train = allIdxs[annot.istrain:eq(1)]

        print(opt.idxRef.train:size()) -- 753
        print(opt.idxRef.test:size())  -- 133

        if not opt.randomValid then
            -- Use same validation set as used in our paper (and same as Tompson et al)
            tmpAnnot = annot.index:cat(annot.person, 2):long()
            
            tmpAnnot:add(-1)

            local validAnnot = hdf5.open(paths.concat(projectDir, 'data/harpe/annot/valid.h5'),'r')
            local tmpValid = validAnnot:read('index'):all():cat(validAnnot:read('person'):all(),2):long()
            print(tempValid)
            opt.idxRef.valid = torch.zeros(tmpValid:size(1))
            opt.nValidImgs = opt.idxRef.valid:size(1)
            print(opt.nValidImgs) -- 133 images in valid
            opt.idxRef.train = torch.zeros(opt.idxRef.train:size(1) - opt.nValidImgs)
            print(opt.idxRef.train:size()) -- 620 images train
            -- Loop through to get proper index values
            local validCount = 1
            local trainCount = 1
            print(annot.index:size(1))
            for i = 1,annot.index:size(1) do
            	-- print(i)
            	-- print(validCount)
            	-- print(trainCount)
            	-- print(validCount <= tmpValid:size(1))
            	-- print(tmpAnnot[i]:equal(tmpValid[validCount]))
            	-- print(1)
                if validCount <= tmpValid:size(1) and tmpAnnot[i]:equal(tmpValid[validCount]) then
                    -- print(2)
                    opt.idxRef.valid[validCount] = i
                    -- print(3)11
                    validCount = validCount + 1
                elseif annot.istrain[i] == 1 then
                	-- print(4)
                	-- print(i)
                	-- print(opt.idxRef.train:size())
                	-- print(trainCount)

                	-- print(i)
                    opt.idxRef.train[trainCount] = i
                    -- print(5)
                    trainCount = trainCount + 1
                end
            end
        else
            -- Set up random training/validation split
            local perm = torch.randperm(opt.idxRef.train:size(1)):long()
            opt.idxRef.valid = opt.idxRef.train:index(1, perm:sub(1,opt.nValidImgs))
            opt.idxRef.train = opt.idxRef.train:index(1, perm:sub(opt.nValidImgs+1,-1))
        end

        torch.save(opt.save .. '/options.t7', opt)
    end

    self.annot = annot
    self.nsamples = {train=opt.idxRef.train:numel(),
                     valid=opt.idxRef.valid:numel(),
                     test=opt.idxRef.test:numel()}

    -- For final predictions
    opt.testIters = self.nsamples.test
    opt.testBatch = 1
end

function Dataset:size(set)
    return self.nsamples[set]
end

function Dataset:getPath(idx)
	-- print("hello")
	-- print(idx)
	--print(ffi.string(self.annot.imgname[1]:char():data()))
    return paths.concat(opt.dataDir,'images',ffi.string(self.annot.imgname[idx]:char():data()))
end

function Dataset:loadImage(idx)
    -- print(self:getPath(idx))
    val = self:getPath(idx)
    print(val)
    val = string.gsub(val, "[^g]*$","")
    print(val)
    return image.load(self:getPath(idx))
end

function Dataset:getPartInfo(idx)
	-- print("GET PART INFO")
	-- print(idx)
    local pts = self.annot.part[idx]:clone()
    local c = self.annot.center[idx]:clone()
    local s = self.annot.scale[idx]
    -- Small adjustment so cropping is less likely to take feet out
    c[2] = c[2] + 15 * s
    s = s * 1.25
    return pts, c, s
end

function Dataset:normalize(idx)
    return self.annot.normalize[idx]
end

return M.Dataset

