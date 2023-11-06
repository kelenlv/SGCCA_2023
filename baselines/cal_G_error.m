function res = cal_G_error(list_view, G, list_U)    
    list_projection = transform(list_view,list_U);
    res = 0;
    for i = 1:numel(list_projection)
        res = res + norm((G - list_projection{i}).^2);
    end
%     p = zeros(size(list_projection{1}));
%     for i = 1:numel(list_projection)
%         v = list_projection{i};
%         p(i, :, :) = v;
%     end
%     res = norm((G - p).^2);
end

function res = transform(list_view, list_U)
    res = cell(1, numel(list_view));
    for i = 1:numel(list_U)
        res{i} = list_view{i} * list_U{i};
    end
end
