function rmse = getRmse(ds, y)
%getRmse evaluate the rmse between the test datastore 'ds' and the
%predicitions of the model 'y'

squaredErrors = 0;
i = 1;
reset(ds);
while(hasdata(ds))
    obs = read(ds);
    obs = obs{1};
    
    pred = y(:,:,:,i);
    
    squaredErrors = squaredErrors + sum((obs(:) - pred(:)) .^ 2);
    
    i = i + 1;
end

rmse = sqrt(squaredErrors / numel(y));

end

