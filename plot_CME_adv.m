width=600;
height=450;
line_width = 2;
marker_size = 10;
font_size = 24;
ROOT = './Data/';
OUTPUT_DIR = [ROOT, 'imgs/'];
if ~exist(OUTPUT_DIR, 'dir')
   mkdir(OUTPUT_DIR)
end
figures = {};
figure_names = {};
curve_styles = {'-v', '-+'};
adv_methods = {'Meta_OFWRG', 'PFOCO_SG'};
adv_method_names = {'Meta OFWRG', 'PFOCO-SG'};
sub_batch_grid = [1, 4, 16, 64];
methods_with_stoch_grad = {'Meta_OFWRG', 'MFW', 'PFOCO_SG'};

colors = [
214,39,40
127,127,127
]./256.0;

dataset = 'toy_adv';
max_num_rounds = 400;
max_per_round_time = 0.28;
max_time = inf;
min_avg_regret = -inf;
max_avg_regret = inf;
max_regret_axis = inf;

num_trials = 6;

color_cell_arr = mat2cell(colors, ones(size(colors, 1), 1), size(colors, 2));
map_adv_method_name = containers.Map(adv_methods, adv_method_names);
map_adv_method_style= containers.Map(adv_methods, curve_styles);
map_adv_method_color = containers.Map(adv_methods, color_cell_arr);

marker_interval = 40;
obj_values_cell_avg = nan;
selected_methods_base = nan;

for k = 1 : length(sub_batch_grid)
    % compute the average result
    if k == 1
        file_name = [ROOT, 'results_', dataset, '_CME_1_1_regret.mat'];
        load(file_name);
        
        obj_values_cell_avg = obj_values_cell;
        selected_methods_base = selected_methods;
        for j = 1 : length(obj_values_cell)
            obj_values_cell_avg{j} = zeros(size(obj_values_cell{j}));
        end
        
        len_cell_arr = length(obj_values_cell);
        for i = 1 : num_trials
            file_name = [ROOT, 'results_', dataset, '_CME_1_', num2str(i), '_regret.mat'];
            load(file_name);
        
            for j = 1 : len_cell_arr
                obj_values_cell_avg{j} = obj_values_cell_avg{j} + obj_values_cell{j} ./ num_trials;
            end
        end
    else
        file_name = [ROOT, 'results_', dataset, '_CME_', num2str(sub_batch_grid(k)), '_1_regret.mat'];
        load(file_name);
        len_cell_arr = length(obj_values_cell);
        for j = 1 : len_cell_arr
            obj_values_cell_avg{j} = obj_values_cell_avg{j} * 0;
        end

        for i = 1 : num_trials
            file_name = [ROOT, 'results_', dataset, '_CME_', num2str(sub_batch_grid(k)), '_', num2str(i), '_regret.mat'];
            load(file_name);
            for j = 1 : len_cell_arr
                target_idx = 0;
                for tmp_idx = 1 : length(obj_values_cell_avg)
                    if strcmp(selected_methods_base{tmp_idx}, selected_methods{j})
                        target_idx = tmp_idx;
                        break;
                    end
                end
                if target_idx == 0
                    error(['no method name ', curr_method]);
                end
                obj_values_cell_avg{target_idx} = obj_values_cell_avg{target_idx} + obj_values_cell{target_idx} ./ num_trials;
            end
        end
    end

    % plot time bar
    bar_width = 0.2;
    the_figure = figure('position', [0, 0, width, height]);
    fig_name =[OUTPUT_DIR, dataset, '_CME_per_round_time_b=', num2str(sub_batch_grid(k))];
    figures{end+1} = the_figure;
    figure_names{end+1} = fig_name;
    
    run_time_per_round = zeros(length(adv_methods), 1);
    
    count = 0;
    labels = cell(size(adv_methods));
    for j = 1 : length(adv_methods)
        curr_method = adv_methods{j};
        target_idx = 0;
        for tmp_idx = 1 : length(selected_methods_base)
            if strcmp(selected_methods_base{tmp_idx}, curr_method)
                target_idx = tmp_idx;
                break;
            end
        end
        if target_idx == 0
            error(['no method name ', curr_method]);
        end
        obj_values = obj_values_cell_avg{target_idx};
        if ismember(curr_method, methods_with_stoch_grad)
            curr_label = [map_adv_method_name(curr_method), ', b=', num2str(sub_batch_grid(k))];
        else
            curr_label = map_adv_method_name(curr_method);
        end
        count = count + 1;
        labels{count} = curr_label;
        run_time_per_round(count) = obj_values(end, 3) / obj_values(end, 1);
    end
    
    
    bar_handles = bar(1:length(labels), diag(run_time_per_round), bar_width, 'stack');
    count = 0;
    for j = 1 : length(adv_methods)
        curr_method = adv_methods{j};
        target_idx = 0;
        for tmp_idx = 1 : length(selected_methods_base)
            if strcmp(selected_methods_base{tmp_idx}, curr_method)
                target_idx = tmp_idx;
                break;
            end
        end
        if target_idx == 0
            error(['no method name ', curr_method]);
        end
        count = count + 1;
        bar_handles(count).FaceColor = 'flat';
        bar_handles(count).CData(count, :) = map_adv_method_color(curr_method);
    end
    
    set(gca, 'FontName', 'Times New Roman');
    set (gca, 'FontSize', font_size);
    xlabel(' '); % to align with the curve plots
    ylabel('average running time per round');
    ylim([0, max_per_round_time]);
    set(gca,'xticklabel', {[]});
    grid on;
    
    hold on;
    hBLG = bar(nan(2, length(adv_methods)));         % the bar object array for legend
    for j = 1 : length(adv_methods)
        curr_method = adv_methods{j};
      hBLG(j).FaceColor = map_adv_method_color(curr_method);
    end
    hLG = legend(hBLG, labels, 'Location', 'NorthEast');








    % plot regret curves
    the_figure = figure('position', [0, 0, width, height]);
    fig_name =[OUTPUT_DIR, dataset, '_CME_regret_b=', num2str(sub_batch_grid(k))];
    figures{end+1} = the_figure;
    figure_names{end+1} = fig_name;

    for j = 1 : length(adv_methods)
        curr_method = adv_methods{j};
        target_idx = 0;
        for tmp_idx = 1 : length(selected_methods_base)
            if strcmp(selected_methods_base{tmp_idx}, curr_method)
                target_idx = tmp_idx;
                break;
            end
        end
        if target_idx == 0
            error(['no method name ', curr_method]);
        end
        obj_values = obj_values_cell_avg{target_idx};
        if ismember(curr_method, methods_with_stoch_grad)
            curr_label = [map_adv_method_name(curr_method), ', b=', num2str(sub_batch_grid(k))];
        else
            curr_label = map_adv_method_name(curr_method);
        end
        curr_curve = plot(obj_values(:, 1), cumsum(obj_values(:, 4)), map_adv_method_style(curr_method), 'DisplayName', curr_label, 'LineWidth', line_width, 'MarkerSize', marker_size, 'MarkerIndices', 1:marker_interval:size(obj_values, 1), 'Color', map_adv_method_color(curr_method));
        curr_curve.MarkerFaceColor = curr_curve.Color;
        hold on;
    end
    
    set(gca, 'FontName', 'Times New Roman');
    set (gca, 'FontSize', font_size);
    xlabel('#rounds');
    ylabel('regret');
    legend('show', 'Location', 'NorthWest');
    xlim([0, max_num_rounds]);
    % ylim([0, max_regret_axis]);
    grid on;
end


for i = 1:length(figures)
    the_figure = figures{i};
    the_name = figure_names{i};
    saveas(the_figure, [the_name, '.eps'], 'epsc');
end
