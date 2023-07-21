function custom_BICGStab(A, b, M, x, tol)
    r = b - A * x
    p = 0
    r_d = r
    ro = 1
    w = 1
    a = 1
    v = 0
    for i = 1:2000
        ro_p = -w * ro
        ro = transpose(r) * r_d
        beta = a * ro / ro_p
        p = r .- beta * (p - w * v)
        p_d = M \ p
        v = A * p_d
        sigma = transpose(v) * r_d
        a = ro / sigma
        s = r - a * v
        s_d = M \ s
        t = A * s_d
        w = (transpose(s) * t) / (transpose(t) * t)
        x = x + a * p_d + w * s_d
        r = s - w * t
        if norm(b - A * x) <= tol
            return x
        end

    end
    return x
end