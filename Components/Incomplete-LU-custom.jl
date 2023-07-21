function custom_iLU(A)
    n = size(A)[1]
    H = zeros(size(A))
    for k = 1:n
        H[k, k] = sqrt(A[k, k])
        for i = k+1:n
            if A[i, k] != 0
                H[i, k] = A[i, k] / A[k, k]
            end
        end
        for j = k+1:n
            for i = j:n
                if A[i, j] != 0
                    H[i, j] = A[i, j] - A[i, k] * A[j, k]
                end
            end
        end
    end
    M = sparse(H * transpose(H))
    return M
end
