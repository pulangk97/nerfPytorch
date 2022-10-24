# %%
import torch
import copy

# copy model lossfunction optim to gpu
# def modeltoGPU(model,loss,learning_rate,numGPU):
#     modellist=[]
#     losslist=[]
#     optimlist=[]
#     if numGPU>torch.cuda.device_count():
#         print("no enough gpu")
#         return 0
#     for i in range(numGPU):
#         modellist.append(copy.deepcopy(model))
#         modellist[i].to(torch.device("cuda:"+str(i)))
#     for i in range(numGPU):
#         losslist.append(copy.deepcopy(loss))
#         losslist[i].to(torch.device("cuda:"+str(i)))
#     for i in range(numGPU):
#         optimlist.append(torch.optim.Adam(modellist[i].parameters(),lr=learning_rate))
#     return modellist,losslist,optimlist
def modeltoGPU(model,loss,learning_rate,numGPU,embeddings=None,if_hash=False):
    modellist=[]
    losslist=[]
    optimlist=[]
    emblist=[]
    if if_hash:
        if numGPU>torch.cuda.device_count():
            print("no enough gpu")
            return 0
        for i in range(numGPU):
            modellist.append(copy.deepcopy(model))
            modellist[i].to(torch.device("cuda:"+str(i)))
        for i in range(numGPU):
            emblist.append(copy.deepcopy(embeddings))
            emblist[i].to(torch.device("cuda:"+str(i)))
        for i in range(numGPU):
            losslist.append(copy.deepcopy(loss))
            losslist[i].to(torch.device("cuda:"+str(i)))
        for i in range(numGPU):
                    # print("debug")
                    optimlist.append(torch.optim.RAdam([
                            {'params': list(modellist[i].parameters()), 'weight_decay': 1e-6},
                            {'params': list(emblist[i].parameters()), 'eps': 1e-15}
                        ], lr=5e-4, betas=(0.9, 0.99)) )   

    else:
        if numGPU>torch.cuda.device_count():
            print("no enough gpu")
            return 0
        for i in range(numGPU):
            modellist.append(copy.deepcopy(model))
            modellist[i].to(torch.device("cuda:"+str(i)))
        for i in range(numGPU):
            losslist.append(copy.deepcopy(loss))
            losslist[i].to(torch.device("cuda:"+str(i)))
        for i in range(numGPU):
            optimlist.append(torch.optim.Adam(modellist[i].parameters(),lr=learning_rate,betas=(0.9, 0.999))) 
    return [modellist,emblist],losslist,optimlist


# copy datalist to gpus
def datatoGPU(data,numGPU):
    output=[]
    if numGPU>torch.cuda.device_count():
        print("no enough gpu")
    for i in range(numGPU):
        output.append(data[i].to(torch.device("cuda:"+str(i))))
    return output

# split img by patch to a list
def splitImgToPatch(data,numGPU):
    numpatch=torch.ceil(torch.tensor(data.shape[-1])/numGPU).to(torch.int)
    output=[]
    splitH=[]
    splitW=[]
    for i in range(numGPU):
        output.append(data[...,i*numpatch:((i+1)*numpatch if (i+1)*numpatch<data.shape[-1] else data.shape[-1])])
        splitH.append(torch.tensor(output[i].shape[-2]))
        splitW.append(torch.tensor(output[i].shape[-1]))
    return [output,splitH,splitW]



# mean params from a modellist 
# return a param list 
# def meanParams(modellist):
#     def getParams(model):
#         params=[]
#         for para in model.parameters():
#             params.append(para.to(torch.device("cpu")))
#         return params
#     totalparam=[]
#     for model in modellist:
#         totalparam.append(getParams(model))
#     nummodel=len(totalparam)
#     numparam=len(totalparam[0])
#     meanparam=[]
#     for i in range(numparam):
#         summodel=0
#         for j in range(nummodel):
#             summodel=summodel+totalparam[j][i]
#         meanparam.append(summodel/nummodel)
#     return meanparam

    
def meanParams(modellist):
    def getParams(model):
        params=[]
        paramsgrad=[]
        for para in model.parameters():
            params.append(para.to(torch.device("cpu")))
            paramsgrad.append(para.grad.to(torch.device("cpu")))
        return params,paramsgrad
    totalparam=[]
    totalparamgrad=[]
    for model in modellist:
        oneparam,onegrad=getParams(model)
        totalparam.append(oneparam)
        totalparamgrad.append(onegrad)
    nummodel=len(totalparam)
    numparam=len(totalparam[0])

    meanparam=[]
    for i in range(numparam):
        summodel=0
        sumgrad=0
        for j in range(nummodel):
            summodel=summodel+totalparam[j][i]
            sumgrad=sumgrad+totalparamgrad[j][i]
        meanparam.append(torch.nn.parameter.Parameter(summodel/nummodel))
        meanparam[i].grad=sumgrad/nummodel
        
    return meanparam

    
# copy mean params grad to gpu's models 
def sendToGPU(modellist,paramlist):
    with torch.no_grad():
        for i in range(len(modellist)):
            n=0
            for para in modellist[i].parameters():
                # para.copy_(paramlist[n])
                para.grad.copy_(paramlist[n].grad)
                n=n+1
    
    





