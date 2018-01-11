function setup(varargin)
% SETUP  Add the required search paths to MATLAB
run matconvnet/matlab/vl_setupnn ;
run vlfeat/toolbox/vl_setup ;

opts.useGpu = false ;
opts.verbose = false ;
opts.enableImReadJPEG = false ;
opts = vl_argparse(opts, varargin) ;

try
  vl_nnconv(single(1),single(1),[]) ;
catch
  warning('VL_NNCONV() does not seem to be compiled. Trying to compile it now.') ;
  vl_compilenn('enableGpu', opts.useGpu, ...
               'enableImReadJPEG', opts.enableImReadJPEG, ...
               'verbose', opts.verbose) ;
end

if opts.useGpu
  try
    vl_nnconv(gpuArray(single(1)),gpuArray(single(1)),[]) ;
  catch
    vl_compilenn('enableGpu', opts.useGpu, ...
                 'enableImReadJPEG', opts.enableImReadJPEG, ...
                 'verbose', opts.verbose) ;
    warning('GPU support does not seem to be compiled in MatConvNet. Trying to compile it now') ;
  end
end
