function accuracy = cal_acc(list_projection)
    v1 = list_projection{1};
    v2 = list_projection{2};
    N = size(v1,1);
    % ªÒ»°±Í«©
    label = unique(v2, 'rows', 'stable');

    res = zeros(N, 1);
    for i = 1:N
        for j = 1:size(label, 1)
            if isequal(v2(i, :), label(j, :))
                res(i) = j;
                break;
            end
        end
    end

    c = 0;
    for i = 1:N
        temp = zeros(N, 2);
        for j = 1:N
            dist = sum((v1(i, :) - v2(j, :)).^2);
            temp(j, :) = [dist, j];
        end
        temp = sortrows(temp, 1, 'ascend'); % …˝–Ú≈≈–Ú
        for iz = 1:size(label, 1)
            tt = v2(temp(1, 2), :);
            if isequal(tt, label(iz, :))
                if iz == res(i)
                    c = c + 1;
                end
            end
        end
    end

    accuracy = c / N;
end
