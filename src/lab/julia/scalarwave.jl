#------------------------------------------------------------------
# Julia implementation of the scalar wave equation. 
# Soham 9 2017
#------------------------------------------------------------------

function cheb(N)
    if N==0
        x = 1.0
        D = 0.0
    else
        n  = [0:N;]
        x  = cospi.(n/N)
        c  = vcat([2], ones(N-1), [2]).*(-1.0).^n                      
        X  = repeat(x', outer = [N+1, 1])
        dX = X .- X'
        D  = - reshape(kron(c', 1./c'), N+1, N+1)./(dX + eye(N+1))
        D  = D - diagm(vec(sum(D,1)))
    end    
    return D, x
end

function clencurt(N)
        n  = [0:N;]
        x  = cospi.(n/N)
        w  = vec(zeros(N+1))
        ii = [1:N;]
        v  = ones(N-1) 
        if mod(N,2)==0
            w[1] = w[N+1] = 1.0/(N^2 - 1)
            for k = 1:(N/2)-1
                 v = v - 2*cospi.(2*k*(n[2:N]/N))/(4*k^2-1)
            end
            v = v - cospi.(n[2:N])/(N^2-1)
        else
            w[1] = w[N+1] = 1.0/N^2
            for k = 1:((N-1)/2)
                 v = v - 2*cospi.(2*k*(n[2:N]/N))/(4*k^2-1)
            end
        end
        w[2:N] = 2.0*v/N
        return w
end

# XXX: Generalize this function for arbitrary dimensions
function operator(N)
    D0, t = cheb(N)
    D1, x = cheb(N)
    I  = eye(N+1)
    D  = - kron(I,D0^2) + kron(D1^2, I)

    V = kron(clencurt(N), clencurt(N)')
    W = diagm(vec(V))
    A = W*D

    return A
end

function boundary(N, A)
    D0, t = cheb(N)
    D1, x = cheb(N)

    I  = eye(N+1)
    Dx = kron(I, D0)
    Dt = kron(D1, I)

    BC = zeros(N+1, N+1)
    bb = zeros(N+1, N+1)
    
    BC[N+1, :]= BC[:, N+1] = BC[:, 1] = BC[1, :] = 1

    bb[:, 1]   =  0.0
    bb[1, :]   =  - sinpi.(x)
    bb[:, N+1] =  0.0
    bb[N+1, :] =  0.0

    # FIXME: Don't loop over this. Set them all at once.
    # SEE: Python implementation
    
    for (index, value) in enumerate(BC)
        if value==1
            A[index,:] = zeros((N+1)^2)
            A[index, index] = 1.0
        elseif value==-1
            A[index,:] = Dt[index, :]
        end
    end

   return A, vec(bb)
end

function solve(N, A, b)
    uu = A \ b
    uu = reshape(uu, N+1, N+1)
    return uu
end

function analysis(A)
    return eig(A)
end

function main(N)
    println("Constructing operator")
    A    = @time operator(N)
    println("Setting boundary conditions")
    A, b = @time boundary(N, A)
    println("Solving the system")
    uu   = @time solve(N, A, b)
end

main(3)
