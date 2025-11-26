function add_path()


scriptFullPath = mfilename('fullpath');
scriptDir = fileparts(scriptFullPath);
scriptParentDir = fileparts(scriptDir);

addpath('../Quantize');
addpath(genpath('../Utilities'));
warning off
currentPath = pwd;
%% python-env
addpath('..\Trained_Weights\MWCNN\');
addpath('..\Trained_Weights\DPIR\');
addpath('..\..\Trained_Weights\Restormer\');
addpath('..\Trained_Weights\sigma_estimate\');

MWCNN_path = [scriptParentDir,'\Trained_Weights\MWCNN\'];
DPIR_path = [scriptParentDir,'\Trained_Weights\DPIR\'];
Restormer_path = [scriptParentDir,'\Trained_Weights\Restormer\'];
sigma_estimate_path = [scriptParentDir,'\Trained_Weights\sigma_estimate\'];

if count(py.sys.path, MWCNN_path) == 0
    insert(py.sys.path,int32(0),MWCNN_path);
end
if count(py.sys.path,DPIR_path) == 0
    insert(py.sys.path,int32(0),DPIR_path);
end
if count(py.sys.path,sigma_estimate_path) == 0
    insert(py.sys.path,int32(0),sigma_estimate_path);
end
if count(py.sys.path,Restormer_path) == 0
    insert(py.sys.path,int32(0),Restormer_path);
end

insert(py.sys.path,int32(0),'D:\Software\Anaconda\envs\torch_matlab_new\');  %%% Setting you python path!
setenv('PATH', 'D:\Software\Anaconda\envs\torch_matlab_new\');


py.importlib.reload(py.importlib.import_module('Restormer_denoise_matlab'));
py.importlib.reload(py.importlib.import_module('MWCNN_matlab'));
py.importlib.reload(py.importlib.import_module('DPIR_matlab'));
py.importlib.reload(py.importlib.import_module('Sigma_hat_matlab'));

end