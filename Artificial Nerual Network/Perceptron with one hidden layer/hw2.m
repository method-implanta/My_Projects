trainingSet = csvread("training_set.csv");
validationSet = csvread("validation_set.csv");

trainingSet(:, 1) = normalize(trainingSet(:, 1));
trainingSet(:, 2) = normalize(trainingSet(:, 2));
validationSet(:, 1) = normalize(validationSet(:, 1));
validationSet(:, 2) = normalize(validationSet(:, 2));

M1 = 15;

w1 = randn(M1, 2);
w2 = randn(M1, 1);

t1 = zeros(M1, 1);
t2 = 0;

v0 = zeros(2, 1);
v1 = zeros(M1, 1);
v2 = 0;

c1 = ones(M1, 1);
c2 = 1;
C = 1;
learningRate = 0.005;
vMax = 1e8;

for v = 1:vMax

    mu = randi(10000);
    v0 = transpose(trainingSet(mu, [1 2]) );

    for j = 1:M1
        v1(j) = tanh(w1(j, :) * v0 - t1(j) );
    end
    v2 = tanh(transpose(w2) * v1 - t2);

    c2 = ( (sech(transpose(w2) * v1 - t2) ) ^ 2) * (trainingSet(mu, 3) - v2);
    for j = 1:M1
        c1(j) = c2 * w2(j) * (sech(w1(j, :) * v0 - t1(j) ) ^ 2);
    end

    for m = 1:M1
        for n = 1:2
            w1(m,n) = w1(m,n) + learningRate * c1(m) * v0(n);
        end
    end
    for m = 1:M1
        t1(m) = t1(m) - learningRate * c1(m);
    end

    for n = 1:M1
        w2(n) =  w2(n) + learningRate * c2 * v1(n);
    end
    t2 = t2 - learningRate * c2;

    if rem(v, 1000) == 0
        C = 0;
        for mu = 1:5000
            v0 = transpose(validationSet(mu, [1 2]) );
            for j = 1:M1
                v1(j) = tanh(w1(j, :) * v0 - t1(j) );
            end
            v2 = tanh(transpose(w2) * v1 - t2);
            C = C + (1/10000) * abs(sign(v2) - validationSet(mu, 3));
        end
        disp(C);
    end

    if C < 0.12
        csvwrite('w1.csv', w1);
        csvwrite('w2.csv', w2);
        csvwrite('t1.csv', t1);
        csvwrite('t2.csv', t2);
        break;
    end
end