--- Load up network model or initialize from scratch
--require('mobdebug').start()
paths.dofile('models/' .. opt.netType .. '.lua')


-- Continuing an experiment where it left off
if opt.continue or opt.branch ~= 'none' then
    local prevModel = opt.load .. '/final_model.t7'
    print('==> Loading model from: ' .. prevModel)
    model = torch.load(prevModel)


-- Or a path to previously trained model is provided
elseif opt.loadModel ~= 'none' then
    assert(paths.filep(opt.loadModel), 'File not found: ' .. opt.loadModel)
    print('==> Loading model from!!!!!!: ' .. opt.loadModel)
    modelTrained = torch.load(opt.loadModel)
    -- params, gradParams = modelTrained:getParameters()

      params, gradParams = modelTrained:getParameters()

      -- modelTrained.weight(5)

      -- model = require('weight-init')(modelTrained,'kaiming')

      -- print(params) print(inp.fowardnodes[#list-1].data.modules:parameters())
      --print(modelTrained)
      model = createModel(modelTrained)


    -- clone and pop top
    -- modelClone = modelTrained:clone()
    -- modelPopTop = modelClone:remove()

    -- newbg = modelPopTop.bg.nodes[1]
    -- thisnode = newbg
    -- x3 = newbg

    -- if opt.resetClassifier == 0 then
      -- print(' => Parameter sharing classifier with ' .. 18 .. '-way classifier')
      -- print(modelTrained:type())
      -- print(modelTrained:data())
      -- modelNew = createModel()
      -- print(modelNew:type())
      -- print(modelTrained.fg:roots())
      -- print(modelNew.fg.roots())


    --   model = nn.gModule({modelTrained.forwardnodes[1].data},modelNew())
    --   print("PAST GMODULE")
    --   print(model.fg:roots())

      -- modelNew = modelTrained:clone('weight','bias','gradWeight','gradBias')

      -- midLayer = nn.Sequential()
      -- midLayer = nn.Contiguous()
      --   :add(nn.SpatialBatchNormalization(opt.nFeats))
        -- -- :add(nnlib.ReLu(true))
        -- :add(nnlib.SpatialConvolution(opt.nFeats,opt.nFeats,1,1))
        -- :add(nn.SpatialBatchNormalization(numOut/2))
        -- :add(relu(true))
        -- :add(nnlib.SpatialConvolution(3,3,opt.nFeats,1,1,1,1))
        -- :add(nn.SpatialBatchNormalization(numOut/2))
        -- :add(relu(true))
        -- :add(nnlib.SpatialConvolution(numOut/2,numOut,1,1))

      -- model = nn.Sequential():add(modelNew):add(midLayer):add(modelTrained)
        -- model = nn.Sequential():add(modelNew):add(nn.utils.recursiveType(modelTrained,'torch.FloatTensor'))
        -- model = nn.Sequential():add(midLayer):add(modelTrained)
              -- model = nn.Sequential():add(modelNew()):add(modelTrained())



      -- params, gradParams = model:getParameters()

      -- params:fill(0)

      -- local orig = model:get(#model.modules)
      -- assert(torch.type(orig) == 'nn.Linear',
      --    'expected last layer to be fully connected')

      -- local linear = nn.Linear(orig.weight:size(2), 18)--opt.nClasses)
      -- linear.bias:zero()

      -- model:remove(#model.modules)
      -- model:add(linear:type(opt.tensorType))
   -- end

-- Or we're starting fresh
else
    print('==> Creating model from file: models/' .. opt.netType .. '.lua')
    print(modelArgs)
    model = createModel(modelArgs)
end

-- Criterion (can be set in the opt.task file as well)
if not criterion then
    criterion = nn[opt.crit .. 'Criterion']()
end

if opt.GPU ~= -1 then
    -- Convert model to CUDA
    print('==> Converting model to CUDA')
    model:cuda()
    criterion:cuda()
    
    cudnn.fastest = true
    cudnn.benchmark = true
end
