using MPIMeasurements, MPIReco

include("handsfreeKaczmarz.jl")

"""
Calculates a noise level from empty measurement `bEmpty`.

    NoiseLevel = getNoiseLevel(bEmpty,bgframes,channels)
"""
function getNoiseLevel(bEmpty,bgframes,channels)
    tmp = getMeasurementsFD(bEmpty,true)[:,channels,:,bgframes]
    if length(channels) > 1
        numfreq,numchannels,numpatches,numframes = size(tmp)
        measBG = reshape(permutedims(tmp,[1,3,2,4]),(numfreq*numpatches,numchannels,numframes))
    else
        numchannels = 1
        numfreq,numpatches,numframes = size(tmp)
        measBG = reshape(tmp,(numfreq*numpatches,numchannels,numframes))
    end
    noise = zeros(numfreq*numpatches,numchannels)
    for r = 1:numchannels
        for k=1:numfreq*numpatches
            maxBG = maximum(abs.(measBG[k,r,:]))
            meanBG = mean(measBG[k,r,:])
            noise[k,r] = mean(abs.((measBG[k,r,:] .- meanBG)))#./maxBG))
        end
    end
    return mean(noise)
end

"""
High-Level hands-free reconstruction deriving all parameters from an empty-measurement `bEmpty` (if given).

    c,it,curv = handsfreeReco(bSF::MPIFile, bMeas::MPIFile; bEmpty = nothing, bgFrames = 1, recChannels = [1,2,3], kargs...)
"""
function handsfreeReco(bSF::MPIFile, bMeas::MPIFile;
    bEmpty = nothing,
    bgFrames = 1,
    recChannels = [1,2,3],
    kargs...
    )

    if bEmpty !== nothing
        NoiseLevel=getNoiseLevel(bEmpty,bgFrames,recChannels)
        startλ=1/(NoiseLevel*0.00005)
        SNRbounds=(0.012,0.00031) .* NoiseLevel
    else
        startλ=5.0
        SNRbounds=(60.0,1.5)   
    end

    return handsfreeReco(bSF, bMeas, startλ, SNRbounds; bEmpty = bEmpty, bgFrames = bgFrames, recChannels = recChannels, kargs...)
end

"""
Mid-Level hands-free reconstruction with given SNRbounds (αΘ,αmin) and starting regularization strength αλ.

    c,it,curv = handsfreeReco(bSF::MPIFile, bMeas::MPIFile, startλ::AbstractFloat, SNRbounds::Tuple{AbstractFloat,AbstractFloat}; kargs...)
"""
function handsfreeReco(bSF::MPIFile, bMeas::MPIFile, startλ::AbstractFloat, SNRbounds::Tuple{AbstractFloat,AbstractFloat}; 
            frames = 1:acqNumFrames(bMeas),
            nAverages = 1,
            bEmpty = nothing,
            recChannels = [1,2,3],
            minFreq = 80e3,
            kargs...
        )

    # final (biggest) frequency selection
    finalfreq = filterFrequencies(bSF, minFreq=minFreq, SNRThresh=SNRbounds[2], recChannels = recChannels)

    bgCorrection = bEmpty !== nothing ? true : false

    # load system matrix and grid
    SM, grid = getSF(bSF, finalfreq, nothing, "kaczmarz"; bgcorrection=bgCorrection)

    return handsfreeReco(SM, bMeas, finalfreq; frames = frames, nAverages = nAverages, bEmpty = bEmpty,
    recChannels = recChannels, minFreq = minFreq, startλ = startλ, SNRbounds = SNRbounds, kargs...)
end

"""
Low-Level hands-free reconstruction with already given system matrix and frequency selection.

    c,it,curv = handsfreeReco(SM::Transpose{ComplexF32, Matrix{ComplexF32}}, bMeas::MPIFile, finalfreq::Vector{Int64}; kargs...)
"""
function handsfreeReco(SM::Transpose{ComplexF32, Matrix{ComplexF32}}, bMeas::MPIFile, finalfreq::Vector{Int64}; 
    frames = 1:acqNumFrames(bMeas),
    nAverages = 1,
    numAverages=nAverages,
    bEmpty = nothing,
    bgFrames = 1,
    recChannels = [1,2,3],
    iterBounds = (1,25),
    equalizeIters = false,
    flattenIters = false,    
    startλ = 5.0,
    SNRbounds = (60.0,1.5),
    spectralLeakageCorrection=true,
    bgCorrectionInternal=false,
    numPeriodAverages=1,
    numPeriodGrouping=1
)

bgCorrection = bEmpty !== nothing ? true : false

# load measurements
u = getMeasurementsFD(bMeas,frequencies=finalfreq,frames=frames,numAverages=numAverages,
                        spectralLeakageCorrection=spectralLeakageCorrection,bgCorrection=bgCorrectionInternal,
                        numPeriodAverages=numPeriodAverages,numPeriodGrouping=numPeriodGrouping)
if bgCorrection
uEmpty = getMeasurementsFD(bEmpty,frequencies=finalfreq,frames=bgFrames,numAverages=length(bgFrames),
                            spectralLeakageCorrection=spectralLeakageCorrection,bgCorrection=bgCorrectionInternal,
                            numPeriodAverages=numPeriodAverages,numPeriodGrouping=numPeriodGrouping)
u .-= uEmpty
end

### --------------- ###
### reconstructrion ###
### --------------- ###

# setup
progress = nothing
M,N = size(SM)
L = size(u)[end]
u = reshape(u, M, L)
c = zeros(N,L)
it = zeros(L)
w_it = zeros(L)
curv = [zeros(iterBounds[2]) for i=1:L]

# build up solver
solv = handsfreeKaczmarz(SM, finalfreq; iterbounds=iterBounds, recChannels=recChannels, SNRbounds=SNRbounds, startλ=startλ)

progress===nothing ? p = Progress(L, 1, "Reconstructing data...") : p = progress

if equalizeIters
    for l=1:L
        d,it_l,curv_l = solve(solv, u[:,l])
        curv[l] = curv_l
        it[l] = it_l
        w_it[l] = solv.wanted_iters
        c[:,l] = real( d )
        next!(p)
        if l >= 3
            solv.expected_iters = round(Int,sum(it[l-2:l].*[0.1,0.3,0.6]))
        end
    end
else
    for l=1:L
        d,it_l,curv_l = solve(solv, u[:,l])
        curv[l] = curv_l
        it[l] = it_l
        c[:,l] = real( d )
        next!(p)
    end
end

if flattenIters
    for l=1:L     
        iter=round(Int,mean(it[maximum((1,l-5)):minimum((l+5,L))]))
        solv.iterbounds = (iter,iter)
        solv.expected_iters = iter
        d,it_l,curv_l = solve(solv, u[:,l])
        curv[l] = curv_l
        w_it[l] = it_l
        c[:,l] = real( d )
        next!(p)
    end
    it .= w_it
end

return (c, it, curv)
end