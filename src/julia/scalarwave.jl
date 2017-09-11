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
    (D, x)
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
        (w)
end
