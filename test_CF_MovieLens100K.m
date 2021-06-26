addpath('./Config/CF/');

fprintf(['MovieLens100K regret (1) ...', '\n']);
for test_idx = 1 : 6
   fprintf(['[trial ', num2str(test_idx), ']\n']);
   figure;
   MovieLens100K_1;
   main_CF;
   output_file_name = [DATA_ROOT, 'results_', dataset, '_CF_1_', num2str(test_idx), '_regret.mat'];
   save(output_file_name, 'selected_methods', 'obj_values_cell');
   fprintf('\n\n');
end

clear;

fprintf(['MovieLens100K regret (4) ...', '\n']);
for test_idx = 1 : 6
   fprintf(['[trial ', num2str(test_idx), ']\n']);
   figure;
   MovieLens100K_4;
   main_CF;
   output_file_name = [DATA_ROOT, 'results_', dataset, '_CF_4_', num2str(test_idx), '_regret.mat'];
   save(output_file_name, 'selected_methods', 'obj_values_cell');
   fprintf('\n\n');
end

clear;

fprintf(['MovieLens100K regret (16) ...', '\n']);
for test_idx = 1 : 6
   fprintf(['[trial ', num2str(test_idx), ']\n']);
   figure;
   MovieLens100K_16;
   main_CF;
   output_file_name = [DATA_ROOT, 'results_', dataset, '_CF_16_', num2str(test_idx), '_regret.mat'];
   save(output_file_name, 'selected_methods', 'obj_values_cell');
   fprintf('\n\n');
end

clear;

fprintf(['MovieLens100K regret (64) ...', '\n']);
for test_idx = 1 : 6
   fprintf(['[trial ', num2str(test_idx), ']\n']);
   figure;
   MovieLens100K_64;
   main_CF;
   output_file_name = [DATA_ROOT, 'results_', dataset, '_CF_64_', num2str(test_idx), '_regret.mat'];
   save(output_file_name, 'selected_methods', 'obj_values_cell');
   fprintf('\n\n');
end
