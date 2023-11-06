function res = cal_spare(list_U)
    res = [];
    for i = 1:numel(list_U)
        disp(['info of sparsity: L1 norm of each view: ', num2str(norm(list_U{i}, 1))]);
    end
    for i = 1:numel(list_U)
        u = list_U{i};
        disp(['shape of list_U: ', num2str(size(u, 1)), ' ', num2str(size(u, 2))]);
        res(i) = sum(abs(u) <= 1e-5) / (size(u, 1) * size(u, 2));
        disp(['info of sparsity: zero number: ', num2str(sum(abs(u) <= 1e-5))]);
    end
end

