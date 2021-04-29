% Functions to balance the Dataset. It uses oversampling.
function datastore = balanceDatastore(datastore)
labelCount = countEachLabel(datastore);   
maxNumObservations = max(labelCount{:, 2});
[G,classes] = findgroups(datastore.Labels);
filesAndLabels = splitapply(@(x,y){randReplicateData(x,y,maxNumObservations)},datastore.Files, datastore.Labels,G); 
filesAndLabels = vertcat(filesAndLabels{:,1});
%files = filesAndLabels(:,1);
%labels = classes(filesAndLabels(:,2));
datastore.Files = cellstr(filesAndLabels(:,1));
datastore.Labels = classes(filesAndLabels(:,2));
end

function t = randReplicateData(files, labels, numDesired)
n = numel(files);
ind = randi(n,numDesired,1);
t = [files(ind),labels(ind)];
end