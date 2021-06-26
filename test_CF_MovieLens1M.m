addpath('./Config/CF/');

fprintf(['MovieLens1M regret (64) ...', '\n']);
for test_idx = 1 : 6
   fprintf(['[trial ', num2str(test_idx), ']\n']);
   figure;
   MovieLens1M_64;
   main_CF;
   output_file_name = [DATA_ROOT, 'results_', dataset, '_CF_64_', num2str(test_idx), '_regret.mat'];
   save(output_file_name, 'selected_methods', 'obj_values_cell');
   fprintf('\n\n');
end
