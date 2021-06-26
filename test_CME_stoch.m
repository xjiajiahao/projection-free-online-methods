addpath('./Config/CME/');

fprintf(['smooth regret (1) ...', '\n']);
for test_idx = 1 : 6
   fprintf(['[trial ', num2str(test_idx), ']\n']);
   figure;
   toy_stoch_1;
   main_CME_stoch;
   output_file_name = [DATA_ROOT, 'results_', dataset, '_CME_1_', num2str(test_idx), '_regret.mat'];
   save(output_file_name, 'selected_methods', 'obj_values_cell');
   fprintf('\n\n');
   drawnow;
end

clear;

fprintf(['smooth regret (4) ...', '\n']);
for test_idx = 1 : 6
   fprintf(['[trial ', num2str(test_idx), ']\n']);
   figure;
   toy_stoch_4;
   main_CME_stoch;
   output_file_name = [DATA_ROOT, 'results_', dataset, '_CME_4_', num2str(test_idx), '_regret.mat'];
   save(output_file_name, 'selected_methods', 'obj_values_cell');
   fprintf('\n\n');
end

clear;

fprintf(['smooth regret (16) ...', '\n']);
for test_idx = 1 : 6
   fprintf(['[trial ', num2str(test_idx), ']\n']);
   figure;
   toy_stoch_16;
   main_CME_stoch;
   output_file_name = [DATA_ROOT, 'results_', dataset, '_CME_16_', num2str(test_idx), '_regret.mat'];
   save(output_file_name, 'selected_methods', 'obj_values_cell');
   fprintf('\n\n');
end

clear;

fprintf(['smooth regret (64) ...', '\n']);
for test_idx = 1 : 6
   fprintf(['[trial ', num2str(test_idx), ']\n']);
   figure;
   toy_stoch_64;
   main_CME_stoch;
   output_file_name = [DATA_ROOT, 'results_', dataset, '_CME_', num2str(test_idx), '_regret.mat'];
   save(output_file_name, 'selected_methods', 'obj_values_cell');
   fprintf('\n\n');
end
