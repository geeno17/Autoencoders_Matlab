load('C:\Users\Pierluigi\Desktop\filter8_dim3.mat')

layer = 2;

filters = net.Layers(layer).Weights;
nFilters = net.Layers(layer).NumFilters;
nChannels = net.Layers(layer).NumChannels;
filterSize = net.Layers(layer).FilterSize;

% come "srotolare" i filtri???
% approccio 1
F = zeros(nFilters, prod(filterSize) * nChannels);
for i = 1 : nFilters    
    for j = 1 : nChannels
        f = filters(:,:,j,i);
        F(i, (j - 1) * prod(filterSize) + 1 : (j - 1) * prod(filterSize) + prod(filterSize)) = reshape(f.',1,[]);
    end
end

% approccio 2
F = zeros(nFilters, prod(filterSize) * nChannels);
for i = 1 : nFilters
    for j = 1 : filterSize(1)
       for k = 1 : nChannels
           f = filters(j,:,k,i);
           F(i, (j - 1) * prod(filterSize) + (k - 1) * filterSize(1) + 1 : (j - 1) * prod(filterSize) + (k - 1) * filterSize(1) + filterSize(1)) = f;
       end        
    end
end

% approccio 3
F = zeros(nFilters * nChannels, prod(filterSize));
for i = 1 : nFilters
   for j = 1 : nChannels
      f = filters(:,:,j,i);
      F((i - 1) * nChannels + j, :) = reshape(f.',1,[]);       
   end
end

eigenvalues = eig(F' * F);
minEigenvalue = min(eigenvalues);
maxEigenvalue = max(eigenvalues);

minEigenvalue, maxEigenvalue
