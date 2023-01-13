import gym
import collections
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pyglet

learning_rate = 0.0003
gamma = 0.98
buffer_limit = 50000
batch_size = 32
scoreSum = []
successSum = []
num_iterations = 2000

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)  #initialize replay buffer

    def put(self, transition):  #buffer에 new transition저장
        self.buffer.append(transition)

    def sample(self, n):  #minibatch를 random하게 sampling
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float32), torch.tensor(a_lst), \
               torch.tensor(r_lst, dtype=torch.float32), torch.tensor(s_prime_lst, dtype=torch.float32), \
               torch.tensor(done_mask_lst)  #r값을 float32형태로 줘서 약간 수정함

    def size(self):  #buffer에 저장된 transition의 수
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(2, 256)  #input layer : 2, hidden layer1 : 256
        self.fc2 = nn.Linear(256, 256)  #hidden layer1 : 256, hidden layer2 : 256
        self.fc3 = nn.Linear(256, 3)  #hidden layer2 : 256, output layer : 3

    def forward(self, x):
        x = F.relu(self.fc1(x))  #ReLU함수 input fc1(x)을 x에 대입
        x = F.relu(self.fc2(x))  #ReLU함수 fc2(x)를 다시 x에 대입
        x = self.fc3(x)  #q값이 양수거나 음수일 수 있기 대문에 마지막에는 activation사용안함
        return x  #값 리턴

    def sample_action(self, obs, epsilon):  #epsilon-greedy action sampling
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:  #epsilon보다 작으면
            return random.randint(0,2)  #explorer
        else:  #epsilon 이상이면
            return out.argmax().item()  #exploit

def train(q, q_target, memory, optimizer):   #training function
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)  #q_out중에서 현재 action에 대한 q값만 걸러내는 것

        # DQN
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        #s_prime은 다음 state,거기에대한 큐값들이 나오는데 거기에 max값을 취하는 것이다.

        target = r + gamma * max_q_prime * done_mask   #타겟밸류이다. done_mask=0(terminal state), =1(not terminal state)

        # MSE Loss
        loss = F.mse_loss(q_a, target)

        optimizer.zero_grad()  #gradient
        loss.backward()  #backpropagation
        optimizer.step()  #parameter update

def main():
    env = gym.make('MountainCar-v0')
    q = Qnet()  #main net  뉴럴네트워크의 아키텍처를 갖는 클래스의 인스턴스를 두개만든다. => q, q_target
    q_target = Qnet()  #target
    q_target.load_state_dict(q.state_dict())  #Q에 있는 값을 Q_target으로 copy. Q에있는 parameter와 Q_target의 parameter를 똑같도록 만든다.
    memory = ReplayBuffer()   #replay buffer를 초기화
    success = 0.0
    print_interval = 100
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)  #optimizer를 adam을 쓰겠다. q의 파라미터를 업데이트하겠다.

    for n_epi in range(num_iterations):  #에피소드 2000번
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))  #Linear annealing from 8% to 1%
        #초반에는 exploration 많이 하도록 epsilon을 크게 잡아줘서 exploer시키고, 뒤로 갈수록 작게해줘서 exploit시킨다.
        s = env.reset()
        done = False
        score = 0.0

        while not done:  #에피소드가 끝날때까지의 while

            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, info = env.step(a)  #done=1 터미널, 0은 터미널 아닌 것
            done_mask = 0.0 if done else 1.0  #여기서 done_mask를 계산위해 수정

            if s_prime[1] > 0:
                r = ((s_prime[0]+1.2)*10)**2/10 + 10*s_prime[1]  # position이 오른쪽으로 갈 수록, 속도가 빠를 수록 받는 r이 증가
            else:
                r = ((s_prime[0]+1.2)*10)**2/10 # 그렇지 않으면 상대적으로 적은 r을 갖게된다.
            if s_prime[0] >= 0.5:  # flag 위치가 goal이면
                success += 1  # flag에 닿으면 성공
                print('success')
                r = ((s_prime[0]+1.2)*10)**2/10 + 10*s_prime[1] + 20  # 성공하면 리워드 20을 추가로 받는다.
            memory.put((s,a,r/100.0,s_prime, done_mask)) #r을 백으로 나눈 이유는 러닝할 경우 들어가는 값이 너무 크면 트레이닝하는데 오래걸리므로
            s = s_prime

            score += r
            if done: #에피소드 종료
                scoreSum.append(score)

                break
            env.render()
        if memory.size()>2000:  #메모리 사이즈 이천 넘으면 트레이닝을 시작하겠ㄷ다.
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:  #interval이 지날때마다 Q를 Q_target으로 카피해주는 것이다.
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, scoreSum[n_epi]/print_interval, memory.size(), epsilon*100))
            successSum.append(success)
            score = 0.0
            success = 0.0
    print(success/num_iterations)
    total_scoreSum = 0
    for i in range(len(scoreSum)):
        print("{}번째 score :".format(i), scoreSum[i])
        total_scoreSum += scoreSum[i]
        print("{}번째 average scoreSum :".format(i), total_scoreSum/(len(scoreSum[0:i])+1))
    env.close()

if __name__ == '__main__':
    main()

plt.grid()
plt.plot(range(num_iterations), scoreSum)
plt.title('Mountain')
plt.xlabel('Training Iterations')
plt.ylabel('Reward')
plt.show()