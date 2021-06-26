clear;
addpath('./Config/SVM/');

fprintf(['eeg regret (1) ...', '\n']);
for test_idx = 1 : 6
    fprintf(['[trial ', num2str(test_idx), ']\n']);
    figure;
    eeg_l1_1;
    main_SVM;
    output_file_name = [DATA_ROOT, 'results_', dataset, '_SVM_l1_1_', num2str(test_idx), '_regret.mat'];
    save(output_file_name, 'selected_methods', 'obj_values_cell');
    fprintf('\n\n');
end

clear;

fprintf(['eeg regret (2) ...', '\n']);
for test_idx = 1 : 6
    fprintf(['[trial ', num2str(test_idx), ']\n']);
    figure;
    eeg_l1_2;
    main_SVM;
    output_file_name = [DATA_ROOT, 'results_', dataset, '_SVM_l1_2_', num2str(test_idx), '_regret.mat'];
    save(output_file_name, 'selected_methods', 'obj_values_cell');
    fprintf('\n\n');
end

clear;

fprintf(['eeg regret (4) ...', '\n']);
for test_idx = 1 : 6
    fprintf(['[trial ', num2str(test_idx), ']\n']);
    figure;
    eeg_l1_4;
    main_SVM;
    output_file_name = [DATA_ROOT, 'results_', dataset, '_SVM_l1_4_', num2str(test_idx), '_regret.mat'];
    save(output_file_name, 'selected_methods', 'obj_values_cell');
    fprintf('\n\n');
end

clear;

fprintf(['eeg regret (8) ...', '\n']);
for test_idx = 1 : 6
    fprintf(['[trial ', num2str(test_idx), ']\n']);
    figure;
    eeg_l1_8;
    main_SVM;
    output_file_name = [DATA_ROOT, 'results_', dataset, '_SVM_l1_8_', num2str(test_idx), '_regret.mat'];
    save(output_file_name, 'selected_methods', 'obj_values_cell');
    fprintf('\n\n');
end
