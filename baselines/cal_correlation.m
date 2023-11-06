function res = cal_correlation(list_projection)
    concat_projection = cat(2, list_projection{:});
    corr_array = corrcoef(concat_projection, 'Rows', 'complete');
    res = abs(corr_array(triu(true(size(corr_array)), 1)));
end
%    % Alternatively, you can call the rank_corr function
%     res = rank_corr(corr_array);
function res = rank_corr(corr_array)
        D = floor(size(corr_array, 1) / 2);
        res = [];
        for i = 1:D
            if ~isnan(corr_array(i, i + D))
                res(end+1) = abs(corr_array(i, i + D));
            end
        end
end

