function HMatrices.KernelMatrix(
    operator::To,
    X::Tt,
    Y::Ts,
) where {To<:BEAST.IntegralOperator,Tt<:BEAST.Space,Ts<:BEAST.Space}
    blkasm = BEAST.blockassembler(operator, X, Y)
    function blkassembler(Z, tdata, sdata)
        fill!(Z, 0.0)
        @views store(v, m, n) = (Z[m, n] += v)
        return blkasm(tdata, sdata, store)
    end

    return HMatrices.KernelMatrix{Function,typeof(X),typeof(Y),scalartype(operator)}(
        blkassembler,
        X,
        Y,
    )
end

function Base.size(K::KernelMatrix{Tf,TX,TY,T}) where {Tf,TX<:BEAST.Space,TY<:BEAST.Space,T}
    return length(K.X.pos), length(K.Y.pos)
end

function Base.getindex(
    K::KernelMatrix{Tf,TX,TY,T},
    i::Int,
    j::Int,
) where {Tf,TX<:BEAST.Space,TY<:BEAST.Space,T}
    blk = zeros(T, 1, 1)
    K.f(blk, [i], [j])
    return blk
end

function Base.getindex(
    K::KernelMatrix{Tf,TX,TY,T},
    i,
    j,
) where {Tf,TX<:BEAST.Space,TY<:BEAST.Space,T}
    blk = zeros(T, length(i), length(j))
    K.f(blk, i, j)
    return blk
end

HMatrices.ClusterTree(X::BEAST.Space) = HMatrices.ClusterTree(X.pos)

function HMatrices.assemble_hmatrix(
    K::KernelMatrix{Tf,TX,TY,T};
    rowtree = ClusterTree(K.X),
    coltree = ClusterTree(K.Y),
    adm = StrongAdmissibilityStd(),
    comp = PartialACA(),
    global_index = HMatrices.use_global_index(),
    threads = HMatrices.use_threads(),
    distributed = false,
) where {Tf,TX<:BEAST.Space,TY<:BEAST.Space,T}
    return HMatrices.assemble_hmatrix(
        K,
        rowtree,
        coltree;
        adm = adm,
        comp = comp,
        global_index = global_index,
        threads = threads,
        distributed = distributed,
    )
end

function HMatrices.getblock!(
    out,
    K::HMatrices.PermutedMatrix{KernelMatrix{Tf,TX,TY,Tm},T},
    irange_,
    jrange_,
) where {Tf,TX<:BEAST.Space,TY<:BEAST.Space,Tm,T}
    irange = rangeConverter(irange_, K)
    jrange = rangeConverter(jrange_, K)
    permuted_irange = K.rowperm[irange]
    permuted_jrange = K.colperm[jrange]
    K.data.f(out, permuted_irange, permuted_jrange)
    return out
end

function rangeConverter(range_, K)
    if range_ isa Colon
        range = axes(K, 1)
    elseif range_ isa Int
        range = [range_]
    else
        range = Vector(range_)
    end
    return range
end
