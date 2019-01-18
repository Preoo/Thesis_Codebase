import torch as tc
#import numpy as np

dtype = tc.float
dev = tc.device('cuda' if tc.cuda.is_available() else 'cpu')

N, D_in, H, D_out = 32, 500, 50, 5
x = tc.randn(N,D_in).to(dev)
y = tc.randn(N,D_out).to(dev)

model = tc.nn.Sequential(
    tc.nn.Linear(D_in, H),
    tc.nn.ReLU(),
    tc.nn.Linear(H, D_out)
    ).to(dev)
if tc.cuda.is_available():
    print('Testing cuda usage')
    #model = model.cuda()
loss_f = tc.nn.MSELoss(reduction='sum')

learningrate = 1e-4
optimizer = tc.optim.Adam(model.parameters(),learningrate)

for t in range(5000):
    y_pred = model(x)
    loss = loss_f(y_pred, y)
    if (t % 100 == 0):
        print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#Testing model usage:
with tc.no_grad():
    print('Testing eval:')
    #ones_tensor = np.ones((N, D_in))
    a = model(tc.ones((N,D_in)).to(dev))
    print(a)

