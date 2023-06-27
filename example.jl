using Pkg
Pkg.activate(".")
Pkg.instantiate()

using LazyArtifacts,MPIReco, Plots

include("handsfreeReco.jl")

########
# Data #
########

@info "Load data"
# get calibration and experiment data
pathToMDFStore = artifact"MDFStore"
store = MDFDatasetStore(pathToMDFStore)
study = getStudies(store,"KidneysDilutionSeries")[1]
exp = getExperiments(study)

# load MPIFiles
bEmpty = MPIFile(exp[1]) # background scan
bSF = MPIFile(joinpath(calibdir(store),"1.mdf")) # system matrix

## load measurements
b=[MPIFile(exp[i]) for i=2:7] # measurement

########
# Reco #
########

concentrations = [round(3.05/(2^(i-1)),digits=2) for i=1:6]
bgframes=1:1000

# get Noise Level and System Matrix manually to speed up reconstruction 
NoiseLevel=getNoiseLevel(bEmpty,bgframes,1:3)
finalfreq = filterFrequencies(bSF, minFreq=80e3, SNRThresh=0.00031*NoiseLevel, recChannels = [1,2,3])
SM, grid = getSF(bSF, finalfreq, nothing, "kaczmarz"; bgcorrection=true)

c=[]
it=[]
curv=[]
for l=1:6
    ## This would be the High-Level Reco, but it would take much longer, building up the SM each time
    #result = handsfreeReco(bSF,b[l]; nAverages=1000, bEmpty=bEmpty, bgFrames=bgframes)
    
    ## Low-Level Handsfree Reco
    result = handsfreeReco(SM,b[l],finalfreq; nAverages=1000, bEmpty=bEmpty, bgFrames=bgframes, startλ=1/(NoiseLevel*0.00005), SNRbounds=(0.012,0.00031) .* NoiseLevel)
    
    ## load particle concentration vector and reshape to gridsize (grid.shape)
    push!(c,(x -> x<0.0 ? 0.0 : x).(reshape(result[1],(21,21,24))))
    
    ## load number of iterations of each reco
    push!(it,result[2])

    ## load curvature of L-curve of each reco
    push!(curv,result[3])
end

############
# Plotting #
############

cplots=[]
for l=1:6
    conc = concentrations[l]
    cmax = maximum(c[l])
    push!(cplots,heatmap(
        squeeze(reverse(c[l][:,:,10],dims=1)),
        clim=(0,cmax),cb=false,border=:none,title="c=$conc",
        aspect_ratio=:equal)
    )
end
plot(cplots...)
