function precision = cal_AROC(list_projection)
    % list_projection: [(N, D), (N, D) ... ]
    
    v = cell(1, numel(list_projection));
    for i = 1:numel(list_projection)
        v{i} = list_projection{i};
    end

    N = size(v{1}, 1);
    precision = [];
    
    for i = 1:numel(list_projection)
        for j = i+1:numel(list_projection)
            precision_ = 0;
            
            for ii = 1:N
                temp = zeros(N, 2);
                
                for jj = 1:N
                    dist = sum((v{i}(ii, :) - v{j}(jj, :)).^2);
                    temp(jj, :) = [dist, jj];
                end
                
                temp = sortrows(temp, 1, 'descend'); % Ωµ–Ú≈≈–Ú
                
                index = find(temp(:, 2) == ii, 1);
                precision_ = precision_ + (index + 1) / N;
            end
            
            precision_ = precision_ / N;
            precision(end+1) = precision_;
        end
    end
end
