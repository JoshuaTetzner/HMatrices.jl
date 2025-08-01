using BEAST
using HMatrices
using CompScienceMeshes

Γ = meshsphere(1.0, 0.1)
op = Helmholtz3D.singlelayer()
space = lagrangecxd0(Γ)

A = KernelMatrix(op, space, space);

@time hm = assemble_hmatrix(A);
@time fm = assemble(op, space, space);
