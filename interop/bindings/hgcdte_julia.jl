module HgCdTe
    export Model, HCore, HInst, Eigens

    using Libdl

    libhgcdte = dlopen("./libinterop.dylib")

    make_model = dlsym(libhgcdte, :make_model)
    make_hcore = dlsym(libhgcdte, :make_hcore)
    make_hinst = dlsym(libhgcdte, :make_hinst)
    gen_eigen = dlsym(libhgcdte, :gen_eigen)

    struct Model
        md::Ptr{Cvoid}
    end

    Model(zs::Array{Float64, 1}, xs::Array{Float64}) =
        begin
            len::UInt64 = length(zs)
            @assert length(zs) == length(xs) "Input arrays should have equal length"
            res = ccall(make_model, Ptr{Cvoid}, (UInt64, Ptr{Float64}, Ptr{Float64}), len, zs, xs)
            Model(res)
        end

    struct HCore
        bs::UInt64
        hc::Ptr{Cvoid}
    end

    HCore(md::Model, bsize::UInt64 = 51) = 
        begin
            res = ccall(make_hcore, Ptr{Cvoid}, (Ptr{Cvoid}, UInt64), md.md, bsize)
            HCore(bsize, res)
        end

    struct HInst
        bs::UInt64
        hi::Ptr{Cvoid}
    end

    HInst(hc::HCore, k::Tuple{Float64, Float64}) = 
        begin
            res = ccall(make_hinst, Ptr{Cvoid}, (Ptr{Cvoid}, Float64, Float64), hc.hc, k[1], k[2])
            HInst(hc.bs, res)
        end
    
    Eigens(hi::HInst)::Array{Float64, 1} = 
        begin
            eig = ccall(gen_eigen, Ptr{Float64}, (Ptr{Cvoid}, ), hi.hi)
            res = unsafe_wrap(Array{Float64, 1}, eig, (8 * hi.bs, )) 
            res
        end
end