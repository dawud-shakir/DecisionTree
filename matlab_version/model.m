% dawud cs529
% decision trees

%clc;
%clear vars, warning("clear vars");

examples = readtable('train.csv');

num_examples = 100;

% pretreat


A = table2cell(examples(1:num_examples, 2:end-1)); % Extracting features

A = A(:,2:end); % remove iterative

num_cols = size(A, 2);



features = zeros(num_examples, num_cols);

for col = 1:num_cols
    col % debug
    
    if col<11 %ischar(A{1, col}) % categorical
   
        % need strings for unique
        if ~ischar(A(:,col))
            A(:,col) = cellfun(@num2str, A(:,col), "UniformOutput", false);
        end
        unique_values = unique(A(:, col));
        value_to_number = containers.Map(unique_values, 1:numel(unique_values));
        features(:, col) = cellfun(@(value) value_to_number(value), A(:, col));
    else
        % threshold by mean
        mu = mean(cell2mat(A(:, col)));
       
        features(:, col) = cell2mat(A(:, col)) > mu;
    end
end





target = table2array(examples(1:num_examples,end));                      % target

root=id3(examples,target,features);  
displayTree(root,0)


function displayTree(node, level)


str = [];

str = [str,' [',num2str(node.s(1)),'+, ',num2str(node.s(2)),'-]'];

if ~isempty(node.branch)
    str =  [str,' (', node.branch,') '];
end

str = [str,node.label];
if node.e~=-1
    str = [str,' ',num2str(node.e)];
end

    disp([repmat('-', 1, level),str]);
    for i = 1:length(node.children)
        child = node.children(i);
        displayTree(child, level + 1);
    end
end

function [node] = id3(examples, target, attributes)

if isempty(examples)
    error('examples empty');
end

% create a root node for the tree
node = struct('label',[],'branch',[],'s',zeros(2,1),'e',0,'children',[]);

% stopping conditions

if all(target==1)
    node.label = '+';
    return
end

if all(target==0)
    node.label = '-';
    return
end

if isempty(attributes)
    if mode(target)==1   % most common classifier
        node.label = '+';
    else
        node.label = '-';
    end
    return
end

[e,order,s] = Gain(examples,target,attributes,'entropy');

best_attr = order(1);
    best = attributes(:,best_attr);    % highest information gain in set
    best_e = e(best_attr);
    %best_label = attributes.Properties.VariableNames{best_attr}; 
    best_label = ""; % debug
    attributes(:,best_attr) = [];

values = unique(best);
 
node.label = best_label;
node.e = best_e;

   
    for j=1:size(values,1)
        %v = strcmp(best,values(j));
        v = find(best == values(j));
  
        child = id3(examples(v,:),target(v,:),attributes(v,:));
        child.s = s{best_attr}(j,:);
        child.branch = values(j);
    
        node.children = [node.children,child];
       
    end
    

    e_s = [];
    for i=1:length(node.children)
        e_s(i) = node.children(i).e;
     end

    [~,idx]=sort(e_s,'descend');
    node.children=node.children(idx);
    

%end

end



function H = Gini_Index(p)
    H = sum(p.*(1-p));
end

function H = Misclassification(p)
    H = 1-max(p);
end

function H = Entropy(p)
    
     % 0*log(0)=0
    
   H = -sum(p(p~=0).*log2(p(p~=0)));
    
end

% calculate information gain
function [gain,order,s] = Gain(examples,target,attributes,purity_test)

if isempty(attributes)
    gain = [];
    order = [];
    return;
end

%positive = sum(strcmp(examples{:,end},'Yes')); 
%negative = sum(strcmp(examples{:,end},'No')); 

positive = sum(target);  
negative = sum(~target); 

S_total = positive + negative;


P = [positive/S_total;negative/S_total];

I = Entropy(P); % expected information


num_attributes = size(attributes,2);
gain = zeros(num_attributes,1);
s = {};
for i=1:num_attributes

    A = attributes(:,i)

    Values = unique(A);

    gain(i) = I;

    s2=[];
    for j=1:size(Values,1)

        v = strcmp(A,Values(j,:));

        p = sum(v&(target==1));
        n = sum(v&(target==0));

        s2 = [s2; [p,n]];

        switch lower(purity_test)
            case 'entropy'
                gain(i) = gain(i) - ((p+n)/S_total)*Entropy([p/(p+n);n/(p+n)]);
            case 'gini'
                gain(i) = gain(i) - ((p+n)/S_total)*Gini_Index([p/(p+n);n/(p+n)]);
            case 'misclassification'
                gain(i) = gain(i) - ((p+n)/S_total)*Misclassification([p/(p+n);n/(p+n)]);
            otherwise
                error(['purity test ',purity_test,' not found'])
        end
    end

s(i) = {s2};

end

[gain,order] = sort(gain,'descend');
s = s(order);

end



