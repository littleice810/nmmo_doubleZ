import bisect
import copy
import heapq
import math
import random
import numpy as np

from ijcai2022nmmo import Team
from ijcai2022nmmo import RollOut, CompetitionConfig, scripted
from nmmo.io import action
from nmmo import config


class MyTeam(Team):

    def __init__(self, team_id: str, env_config: config.Config, **kwargs):
        # reset some states
        super().__init__(team_id, env_config, **kwargs)
        self.agents=[MyAgent(i) for i in range(1,9)]
        self.end_explore=False
        self.end_forging=False
        self.time_step=1
        self.team_id=team_id
        self.config=env_config
        self.last_pos=(-100,-100)
    def act(self, observations: dict[int, dict]) -> dict[int, dict]:
        attacker_info,near_foods, roles,near_enemies=self.get_team_info(observations)
        actions = {}
        next_pos= {}
        for player_idx, obs in observations.items():
            if self.agents[player_idx].attacking or self.agents[player_idx].role=='hider':
                tmp_pos=(obs['Entity']['Continuous'][0][5],obs['Entity']['Continuous'][0][6])
                next_pos[tmp_pos] = tmp_pos
        for player_idx, obs in observations.items():
            if self.agents[player_idx].target_pos==(80.1,80.1) and player_idx in [1,2,3,4,5,6,7]:
                cur_agent=obs['Entity']['Continuous'][0]
                cur_pos=(cur_agent[5],cur_agent[6])
                last_agent=observations[player_idx-1]['Entity']['Continuous'][0]
                last_agent_pos=(last_agent[5],last_agent[6])

                if cur_pos[0] in [16,144]:
                    if cur_pos[1]>last_agent_pos[1]:
                        self.agents[player_idx].target_pos = (80, 50 + 20 * (player_idx-1))
                    else:
                        self.agents[player_idx].target_pos = (80, 110 - 20 * (player_idx-1))
                else:
                    if cur_pos[0] > last_agent_pos[0]:
                        self.agents[player_idx].target_pos = (50 + 20 * (player_idx - 1),80)
                    else:
                        self.agents[player_idx].target_pos = (110 - 20 * (player_idx - 1),80)
            actions[player_idx],next_p = self.agents[player_idx].step(obs,attacker_info,near_foods,roles,near_enemies,next_pos)
            if not self.agents[player_idx].freezed:
                next_pos[self.agents[player_idx].position]=next_p
            else:
                next_pos[self.agents[player_idx].position]=self.agents[player_idx].position
            if self.agents[player_idx].end_explore:
                self.end_explore=True
            if self.end_forging:
                self.agents[player_idx].end_forging=True
            else:
                if self.agents[player_idx].end_forging:
                    self.end_forging=True
        if self.last_pos[0]>0 and self.agents[0].get_linf(self.last_pos,self.agents[0].position)>1 :
            self.__init__(self.team_id,self.config)
        self.last_pos = self.agents[0].position
        return actions
    def get_team_info(self,observations):
        attacker_info=[] #协作，如果队友被攻击，距离近则支援
        agents_info=[]
        near_enemies={}
        near_foods={}
        self.agents[0].leader =None
        self.agents[1].leader =None
        self.agents[6].leader =None
        self.agents[7].leader =None
        for player_idx, obs in observations.items():
            all_entity=obs['Entity']['Continuous']
            cur_agent_info=all_entity[0]
            agents_info.append(cur_agent_info)

            if not self.end_explore:
                if cur_agent_info[1] == self.agents[0].id:
                    self.agents[1].leader=cur_agent_info
                if cur_agent_info[1] == self.agents[7].id:
                    self.agents[6].leader=cur_agent_info
            #获取攻击者信息
            cur_attacker_id=cur_agent_info[2]
            if cur_attacker_id != 0:
                attacker=[x for x in all_entity if x[1]  == cur_attacker_id]
                if attacker==[]:
                    continue
                else:
                    attacker=attacker[0]
                dis=max(abs(cur_agent_info[5] - attacker[5]), abs(cur_agent_info[6] - attacker[6]))
                if dis<=1:
                    damage=(7+0.7*cur_agent_info[3]+1)/2
                elif dis<=3 :
                    damage=(3+0.3*cur_agent_info[3]+1)/2
                else:
                    damage=(1+0.3*cur_agent_info[3]+1)/2
                rest_round=min(cur_agent_info[11],attacker[11])/damage
                attacker_info.append((attacker,cur_agent_info,rest_round))

            team_id=cur_agent_info[4]
            near_enemies[(cur_agent_info[5],cur_agent_info[6])] = []
            for entity in all_entity:
                if entity[0] and entity[4]>0 and entity[4]!=team_id:
                    near_enemies[(cur_agent_info[5],cur_agent_info[6])].append(entity)

            all_tile=obs['Tile']['Continuous']
            min_dis=99
            near_food=None
            for tile in all_tile:
                if tile[1]!=4:
                    continue
                position = (tile[2], tile[3])
                if (cur_agent_info[5],cur_agent_info[6]) == position or cur_agent_info[9]>5 or cur_agent_info[12]:
                    near_food = None
                    break
                dis=self.agents[0].get_l1((cur_agent_info[5],cur_agent_info[6]),position)
                if dis<min_dis and position not in near_foods.values():
                    min_dis=dis
                    near_food=position
            if near_food is not None:
                near_foods[(cur_agent_info[5],cur_agent_info[6])]=near_food

        roles={}
        if self.end_explore:
            killed_enemy=[]
            for agent in agents_info:
                for self_agent in self.agents:
                    if self_agent.id==agent[1]:
                        killed_enemy.append(len(self_agent.kill_enemy)+self_agent.level/100)
                        break
            if len(killed_enemy)==1:
                roles[agents_info[0][1]]='hider'
            else:
                ranked_killnum=sorted(killed_enemy,reverse=True)
                for i in range(len(killed_enemy)-1):
                    cur_agent = agents_info[killed_enemy.index(ranked_killnum[i])]
                    roles[cur_agent[1]] = 'attacker'
                cur_agent = agents_info[killed_enemy.index(ranked_killnum[-1])]
                roles[cur_agent[1]]='hider'
        # else:
        #     levels = [agents_info[i][3] + i * 0.01 for i in range(len(agents_info))]
        #     ranked_levels = sorted(levels, reverse=True)
        #     if len(levels) > 2 and ranked_levels[1] > 15:
        #         cur_agent = agents_info[levels.index(ranked_levels[0])]
        #         roles[cur_agent[1]] = 'explorer'
        #         cur_agent = agents_info[levels.index(ranked_levels[1])]
        #         roles[cur_agent[1]] = 'explorer'
        #     elif ranked_levels[0] > 15:
        #         cur_agent = agents_info[levels.index(ranked_levels[0])]
        #         roles[cur_agent[1]] = 'explorer'

        if self.agents[0].explore_dir is None:
            if agents_info[0][5]==agents_info[1][5]:
                if agents_info[0][6]>agents_info[1][6]:
                    self.agents[0].explore_dir='c_max'
                    self.agents[1].explore_dir = 'c_max'
                    self.agents[2].explore_dir = 'c_max_c'
                    self.agents[3].explore_dir = 'c_max_c'
                    self.agents[4].explore_dir = 'c_min_c'
                    self.agents[5].explore_dir = 'c_min_c'
                    self.agents[6].explore_dir = 'c_min'
                    self.agents[7].explore_dir = 'c_min'
                else:
                    self.agents[0].explore_dir='c_min'
                    self.agents[1].explore_dir = 'c_min'
                    self.agents[2].explore_dir = 'c_min_c'
                    self.agents[3].explore_dir = 'c_min_c'
                    self.agents[4].explore_dir = 'c_max_c'
                    self.agents[5].explore_dir = 'c_max_c'
                    self.agents[6].explore_dir = 'c_max'
                    self.agents[7].explore_dir = 'c_max'
            else:
                if agents_info[0][5]>agents_info[1][5]:
                    self.agents[0].explore_dir='r_max'
                    self.agents[1].explore_dir = 'r_max'
                    self.agents[2].explore_dir = 'r_max_c'
                    self.agents[3].explore_dir = 'r_max_c'
                    self.agents[4].explore_dir = 'r_min_c'
                    self.agents[5].explore_dir = 'r_min_c'
                    self.agents[6].explore_dir = 'r_min'
                    self.agents[7].explore_dir = 'r_min'
                else:
                    self.agents[0].explore_dir='r_min'
                    self.agents[1].explore_dir = 'r_min'
                    self.agents[2].explore_dir = 'r_min_c'
                    self.agents[3].explore_dir = 'r_min_c'
                    self.agents[4].explore_dir = 'r_max_c'
                    self.agents[5].explore_dir = 'r_max_c'
                    self.agents[6].explore_dir = 'r_max'
                    self.agents[7].explore_dir = 'r_max'
        return attacker_info,near_foods,roles,near_enemies


class MyAgent():
    def __init__(self,i):
        self.config=config
        self.id=i
        self.obstacle=[]
        self.waters=[]
        self.team=0
        self.position=(0,0)
        self.min_r = 128+16
        self.max_r = 16
        self.min_c = 128+16
        self.max_c = 16
        self.food_carried=0
        self.water_carried=0
        self.food_max=10
        self.water_max=10
        self.health_max=10
        self.path=None
        self.role=None
        self.end_explore=False
        self.end_forging=False
        self.dir = 1
        self.explore_dir=None
        self.visited=[]
        self.exp_new=False
        self.level=1
        self.searched= {} #记录经过的点，用于避免重复绕圈
        self.attacking=None
        self.attacking_info=None
        self.info=None
        self.evading = False
        self.leader=None
        self.freezed=False
        self.dir_list=[]
        self.cur_des=None
        self.init_pos=None
        self.kill_npc=[]
        self.kill_enemy=[]
        self.target_pos=(80.1,80.1)
        x=20+5*(self.id%4)
        x1=140-5*(self.id%4)
        self.target_list=[(x,x),(x1,x),(x,x1),(x1,x1),(x,x),(x1,x),(x,x1),(x1,x1),
                          (x,x),(x1,x),(x,x1),(x1,x1),(x,x),(x1,x),(x,x1),(x1,x1)]
        if self.id%2:
            self.target_list=[(x,x1),(x,x),(x1,x1),(x1,x),(x,x1),(x,x),(x1,x1),(x1,x),
                              (x,x1),(x,x),(x1,x1),(x1,x),(x,x1),(x,x),(x1,x1),(x1,x)]
        self.PARENT = None
        self.g = None
        self.OPEN = None
        self.end = None
        self.end_type = None
        self.attack_pos=None
        self.team_info=[]

    def step(self,obs,attacker_info,near_foods,roles,near_enemies,next_pos):
        all_entity=obs['Entity']['Continuous']
        all_tile=obs['Tile']['Continuous']
        self.attack_pos = None
        self.get_info(all_entity,all_tile,near_foods,next_pos)
        if roles and self.id in roles.keys():
            self.role=roles[self.id]
        else:
            self.role=None
        if self.role=='hider' and self.end_forging:
            self.role='attacker'
        direction, attack_style, target_id = self.help(attacker_info)
        # if not direction:
        #     direction =self.team_work(near_enemies)
        if not direction:
            direction=self.evade()
        if not direction:
            direction = self.find_near_res()

        direction_tmp = None
        if target_id is None:
            direction_tmp, attack_style, target_id = self.attack()
        if not direction and direction_tmp:
            direction = direction_tmp

        if not direction:
            direction=self.attack_npc()
        if not direction:
            direction=self.explore()
        if self.attack_pos is not None and \
                self.get_linf((self.position[0]+direction[0],self.position[1]+direction[1]),self.attack_pos)>4 : # 只有npc为-1时才会触发
            direction=(0,0)
        direction1=-1
        if direction == (-1, 0):
            direction1= 0
        elif direction == (1, 0):
            direction1= 1
        elif direction == (0, 1):
            direction1 = 2
        elif direction == (0, -1):
            direction1= 3

        self.visited.append((self.position[0]+direction[0],self.position[1]+direction[1]))
        if len(self.visited)>30:
            self.visited.pop(0)
        # print(direction)
        if direction1>=0:
            if target_id:
                target=[x for x in range(all_entity.shape[0]) if all_entity[x][1]  == target_id][0]
                self.attacking=target_id
                self.attacking_info=[all_entity[x] for x in range(all_entity.shape[0]) if all_entity[x][1]  == target_id][0]
                actions ={action.Attack: {action.Style: attack_style,
                                        action.Target: target},
                        action.Move: {action.Direction: direction1}}
            else:
                self.attacking=None
                self.attacking_info = None
                actions ={action.Move: {action.Direction: direction1}}
        else:
            if target_id:
                target=[x for x in range(all_entity.shape[0]) if all_entity[x][1]  == target_id][0]
                self.attacking=target_id
                self.attacking_info=[all_entity[x] for x in range(all_entity.shape[0]) if all_entity[x][1]  == target_id][0]
                actions ={action.Attack: {action.Style: attack_style,
                                        action.Target: target}}
            else:
                self.attacking = None
                self.attacking_info = None
                actions={}
        return actions,(self.position[0]+direction[0],self.position[1]+direction[1])

    def help(self,attacker_info):
        direction= None
        attack_style=2
        target_id=None
        min_dis=999
        target=None

        for (attacker,team_mate,rest_round) in attacker_info:
            if self.win_rate(attacker)<0 or attacker[4]<0:
                continue
            dis_l1=self.get_l1(self.position,(attacker[5],attacker[6]))
            dis_linf=self.get_linf(self.position,(attacker[5],attacker[6]))
            if dis_linf<=4:
                dis_teammate_enemy=self.get_linf((attacker[5],attacker[6]),(team_mate[5],team_mate[6]))
                if dis_linf>=dis_teammate_enemy and self.health>team_mate[11]:
                    target = attacker
                    break
                else:
                    direction_tmp, attack_style, target_id = self.attack()
                    return (0,0),attack_style,target_id
            if dis_l1-dis_linf>4:
                dis=dis_l1-8
            else:
                dis=max(0,dis_linf-4)
            if dis+2<=min(7,rest_round) and self.water_carried>dis and self.food_carried>dis and self.win_rate(attacker)+(rest_round-dis)/10>0.6:
                if dis <min_dis:
                    min_dis=dis
                    target=attacker


        if target is not None:
            dis, path = self.dijkstra('end', (target[5], target[6]))
            direction = (path[0][0] - self.position[0], path[0][1] - self.position[1])
            # path.pop(0)
            # self.path = path
            # target_id=target[1]
            return direction,attack_style,target_id
        return direction,attack_style,target_id

    def team_work(self,near_enemies):
        if self.level<10 or self.attacking:
            return None
        target = None
        direction=None
        min_dis=100
        for teammate_pos,enemies in near_enemies.items():
            for enemy in enemies:
                if self.win_rate(enemy) < 0.2 or self.food_carried<5 or self.water_carried<5:
                    continue
                teammate_enemy_dis=self.get_attack_dis(teammate_pos, (enemy[5], enemy[6]))
                self_enemy_dis=self.get_attack_dis(self.position, (enemy[5], enemy[6]))
                if teammate_enemy_dis+5>self_enemy_dis>teammate_enemy_dis and teammate_enemy_dis<=5:
                    if teammate_enemy_dis<min_dis:
                        target=enemy
                        min_dis=teammate_enemy_dis
                    if teammate_enemy_dis<=0:
                        target = enemy
                        break
                    if self_enemy_dis<=2 and teammate_enemy_dis<=2 and self.health>0.8*self.health_max:
                        target = enemy
                        break
        if target is not None:
            dis, path = self.dijkstra('end', (target[5], target[6]))
            direction = (path[0][0] - self.position[0], path[0][1] - self.position[1])
        return direction

    def get_info(self,all_entity,all_tile,near_foods,next_pos):
        self.forest=[]
        self.waters=set()
        self.obstacle_entity=[]
        self.enemies = []
        self.NPCs=[]
        self.obstacle_foods=[]

        for i in range(len(all_entity)):
            entity=all_entity[i]
            if i==0:
                self.info=entity
                self.id=entity[1]
                self.position = (entity[5], entity[6])
                if self.init_pos is None:
                    self.init_pos=self.position
                elif self.get_linf(self.position,self.init_pos)==128:
                    self.end_explore=True
                self.min_r = min(self.min_r, self.position[0])
                self.max_r = max(self.max_r, self.position[0])
                self.min_c = min(self.min_c, self.position[1])
                self.max_c = max(self.max_c, self.position[1])
                self.attacker=entity[2]
                self.level=entity[3]
                self.team = entity[4]
                self.food_carried = entity[9]
                self.water_carried = entity[10]
                self.health=entity[11]
                self.freezed=entity[12]
                self.health_max=max(self.health_max,self.health)
                self.food_max = max(self.food_max, self.food_carried)
                self.water_max = max(self.water_max, self.water_carried)
                if self.food_max+self.water_max>=100:
                    self.end_forging=True
            elif entity[0]:

                if entity[4]==self.team:
                    continue
                elif entity[4]<0:
                    self.NPCs.append(entity)
                else:
                    self.enemies.append(entity)
                if self.get_l1(self.position, (entity[5], entity[6])) == 1:
                    self.obstacle_entity.append((entity[5], entity[6]))
        for cur_p,next_p in next_pos.items():
            if self.get_l1(self.position, next_p) == 1:
                self.obstacle_entity.append(next_p)
            if self.get_l1(self.position, next_p) == 0:
                self.obstacle_entity.append(cur_p)
        i=0
        while i<len(self.obstacle):
            if self.get_linf(self.position,self.obstacle[i])>30:
                self.obstacle.pop(i)
                i-=1
            i+=1

        for tile in all_tile:
            position=(tile[2],tile[3])
            if tile[1] in [0,1,5] and position not in self.obstacle: #熔岩，水，石头
                self.obstacle.append(position)
            if tile[1]==1: #用集合计算提高速度
                neighbor_set=set(self.get_neighbor(position))
                self.waters=self.waters.union(neighbor_set)
            elif tile[1]==4 and (tile[0]==0 or self.position ==position):
                self.forest.append(position)
                if self.position ==position:
                    self.food_carried=self.food_max
        self.waters=list(self.waters.difference(set(self.obstacle)))


        for agent_pos,food_pos in near_foods.items():
            if agent_pos==self.position or self.food_carried<=2:
                continue
            if food_pos in self.forest:
                self.obstacle_foods.append(food_pos)
                self.forest.remove(food_pos)


    def evade(self):
        direction= None
        danger_dir=self.get_danger_dir()
        if danger_dir is not None:
            dis, path = self.dijkstra('des',danger_dir)
            direction=(path[0][0]-self.position[0],path[0][1]-self.position[1])
            return direction
        for npc in self.NPCs:
            if self.get_linf((npc[5],npc[6]),self.position)<=4 and self.get_l1((npc[5],npc[6]),self.position)>4 and int(npc[1])==self.attacking:
                continue
            if int(npc[4]) ==-3 and self.win_rate(npc)<0.6 and self.get_l1((npc[5],npc[6]),self.position)<=5:
                for neighbor in self.get_neighbor(self.position):
                    if self.get_l1(neighbor,(npc[5],npc[6]))<self.get_l1(self.position,(npc[5],npc[6])) and \
                            ((npc[5],npc[6])) not in self.obstacle_entity:
                        self.obstacle_entity.append((npc[5],npc[6]))
            if int(npc[4]) == -3 and self.get_l1((npc[5], npc[6]), self.position) <= 4:
                dis, path = self.dijkstra('evade_npc', (npc[5], npc[6]))
                direction = (path[0][0] - self.position[0], path[0][1] - self.position[1])
                return direction
        return direction
    def find_near_res(self):
        direction=None
        if  self.forest and self.position not in self.forest and self.food_carried<max(7,int(0.6*self.food_max)) and self.food_carried<=self.water_carried:
            for food in self.forest:
                if self.get_l1(self.position,food)<10:
                    dis, path = self.dijkstra('food',None)
                    direction = (path[0][0] - self.position[0], path[0][1] - self.position[1])
                    return direction
        elif self.waters and self.position not in self.waters and self.water_carried<max(7,int(0.6*self.water_max)):
            dis, path = self.dijkstra('water',None)
            direction = (path[0][0] - self.position[0], path[0][1] - self.position[1])
            return direction
        return direction

    def go_circle(self,start):
        if self.level<15 and self.get_l1(self.position,self.target_pos)>10:
            rep=(self.target_pos[0]-self.position[0]+0.1,self.target_pos[1]-self.position[1]+0.1)
            dir=(int(8*rep[0]/math.sqrt(rep[0]**2+rep[1]**2)),int(8*rep[1]/math.sqrt(rep[0]**2+rep[1]**2)))
            end_pos=(min(144,max(16,self.position[0]+dir[0])),min(144,max(16,self.position[1]+dir[1])))
            return end_pos
        if len(self.kill_npc)<1:
            if self.get_l1(self.position,(80,80))<10:
                end_pos=(random.randint(-20,20)+80,random.randint(-20,20)+80)
                while self.get_l1(self.position,end_pos)<7 or self.get_l1(self.position,end_pos)>16:
                    end_pos = (random.randint(-20, 20) + 80, random.randint(-20, 20) + 80)
            else:
                end_pos = (self.position[0] + random.randint(3,7) * (80.1 - self.position[0]) / abs(80.1 - self.position[0]),
                               self.position[1] + random.randint(3,7) * (80.1 - self.position[1]) / abs(80.1 - self.position[1]))
            return end_pos
        else:
            if self.get_l1(self.position,self.target_list[0])>10:
                target_pos=self.target_list[0]
            else:
                self.target_list.pop(0)
                target_pos=self.target_list[0]

            rep = (target_pos[0] - self.position[0] + 0.1, target_pos[1] - self.position[1] + 0.1)
            dir = (int(8 * rep[0] / math.sqrt(rep[0] ** 2 + rep[1] ** 2)),
                   int(8 * rep[1] / math.sqrt(rep[0] ** 2 + rep[1] ** 2)))
            end_pos = (min(144, max(16, self.position[0] + dir[0])), min(144, max(16, self.position[1] + dir[1])))
            return end_pos

            m=10
            # r=min(50,max(8,75-self.level*3))
            r=50+random.randint(-10,10)
            x0,y0=start[0]-80,start[1]-80
            dis=self.get_eucl((0,0),(x0,y0))
            x=x0*r/(dis+0.01)
            y=y0*r/(dis+0.01)
            y1 = y - y * m * m / 2 / r / r - (x * m / 2 / r / r * math.sqrt(4 * r * r - m * m))*self.dir
            x1 = x - x * m * m / 2 / r / r + (y * m / 2 / r / r * math.sqrt(4 * r * r - m * m))*self.dir
            end_pos = (max(-6, min(6, 80+x1 - self.position[0])) + self.position[0],
                       max(-6, min(6, 80+y1 - self.position[1])) + self.position[1])
            if self.get_linf(end_pos,start)>4:
                end_pos = (max(-6, min(6, 80+x - self.position[0])) + self.position[0],
                       max(-6, min(6, 80+y - self.position[1])) + self.position[1])
            return end_pos

    def explore(self):
        if self.path and self.get_l1(self.position,self.path[0])==1:
            for pos in self.path:
                if pos in self.obstacle or pos in self.obstacle_entity or pos in self.obstacle_foods:
                    break
            else:
                direction = (self.path[0][0] - self.position[0], self.path[0][1] - self.position[1])
                self.path.pop(0)
                return direction

        if self.role=='hider':
            if self.food_carried<=self.food_max*0.85 and self.forest and self.position not in self.forest:
                keyword='food'
            elif self.position in self.waters:
                return (0,0)
            else:
                keyword='water'
            dis, path = self.dijkstra( keyword,None )
            direction = (path[0][0] - self.position[0], path[0][1] - self.position[1])
            return direction

        if self.role=='attacker':
            end_pos =self.go_circle(self.position)
            end_pos = self.get_near_des(end_pos)
            dis, path = self.dijkstra('des', end_pos)
            direction=(path[0][0]-self.position[0],path[0][1]-self.position[1])
            path.pop(0)
            self.path = path
            return direction

        if self.level<20 and self.explore_dir[-1]=='c':
                # end_pos = (max(-6, min(6, 80 - self.position[0])) + self.position[0],
                #        max(-6, min(6, 80 - self.position[1])) + self.position[1])
            end_pos = self.go_circle(self.position)
            end_pos=self.get_near_des(end_pos)
            dis, path = self.dijkstra('des', end_pos)
            direction=(path[0][0]-self.position[0],path[0][1]-self.position[1])
            path.pop(0)
            self.path = path
            return direction
        dmax_r = 144 - self.position[0] if self.max_r != 144 else 999
        dmin_r = self.position[0] - 16 if self.min_r != 16 else 999
        dmax_c = 144 - self.position[1] if self.max_c != 144 else 999
        dmin_c = self.position[1] - 16 if self.min_c != 16 else 999

        if self.explore_dir=='c_max' and self.max_c!=144:
            dmax_c=0
        elif self.explore_dir=='r_max' and self.max_r!=144:
            dmax_r=0
        elif self.explore_dir=='c_min' and self.min_c!=16:
            dmin_c=0
        elif self.explore_dir=='r_min' and self.min_r!=16:
            dmin_r=0

        if self.init_pos[0]==16:
            dmax_r=0.1
        elif self.init_pos[0]==144:
            dmin_r=0.1
        elif self.init_pos[1]==16:
            dmax_c=0.1
        elif self.init_pos[1]==144:
            dmin_c=0.1

        min_dis = min(dmax_c, dmax_r, dmin_c, dmin_r)

        if dmax_r == min_dis:
            end_type = 'r'
            end = min(144, self.position[0] + 7)
        elif dmin_r == min_dis:
            end_type = 'r'
            end = max(16, self.position[0] - 7)
        elif dmax_c == min_dis:
            end_type = 'c'
            end = min(144, self.position[1] + 7)
        else:
            end_type = 'c'
            end = max(16, self.position[1] - 7)
        dis, path = self.dijkstra(end_type, end)
        direction = (path[0][0] - self.position[0], path[0][1] - self.position[1])
        path.pop(0)
        self.path = path
        return direction

    def attack(self):
        attack_style=None
        target_id=None
        target=None
        min_hp=1000
        direction=None

        #优先攻击敌方，然后才是NPC
        for enemy in self.enemies:
            position=(enemy[5],enemy[6])
            linf=self.get_linf(self.position,position)
            if linf<5 and enemy[11]<min_hp: #todo 还要判断是否打得过
                min_hp=enemy[11]
                target=enemy

        if target is not None:
            target_id=int(target[1])
            linf = self.get_linf(self.position, (target[5],target[6]))
            if linf <= 1 and self.level<15:
                attack_style = 0
            elif linf <= 3 and self.level<15:
                attack_style = 1
            else:
                attack_style = 2
            if int(target[1])==self.attacking and target[1] not in self.kill_enemy:
                self.kill_enemy.append(target[1])

            return direction, attack_style, target_id

        #优先攻击正在进攻的
        for npc in self.NPCs:
            position=(npc[5],npc[6])
            linf=self.get_linf(self.position,position)
            l1=self.get_l1(self.position,position)
            if int(npc[1])==self.attacking:
                if npc[3]>=25 and npc[1] not in self.kill_npc:
                    self.kill_npc.append(npc[1])
                target=npc
                break
        #其次是-3NPC
        if target is None:
            for npc in self.NPCs:
                position = (npc[5], npc[6])
                linf = self.get_linf(self.position, position)
                if int(npc[4]) ==-3 and linf<=4:
                    target=npc
                    break
        min_l1 = 10
        if target is None:
            for npc in self.NPCs:
                position = (npc[5], npc[6])
                linf = self.get_linf(self.position, position)
                l1 = self.get_l1(self.position, position)
                if (int(npc[4]) in [-1,-3] or l1>4) and linf<=4 and l1<min_l1:
                    min_l1 = l1
                    target=npc

        if target is not None:
            target_id=int(target[1])
            linf = self.get_linf(self.position, (target[5],target[6]))
            attack_style = 2
            if min_l1==5:
                for neighbor in self.get_neighbor(self.position):
                    if self.get_l1(neighbor, (target[5], target[6])) < 5 and \
                            ((target[5], target[6])) not in self.obstacle_entity:
                        self.obstacle_entity.append((target[5], target[6]))
        return direction,attack_style,target_id

    def attack_npc(self):
        direction=None
        choose_enemy=None
        for npc in self.NPCs:
            if npc[1]==self.attacking and self.get_linf(self.position,(npc[5],npc[6]))<=4:
                tmp_npc = copy.deepcopy(npc)
                tmp_npc[5] = self.position[0] + self.get_l1(self.position, (npc[5],npc[6]))- 1
                tmp_npc[6] = self.position[1]
                if int(npc[4]) == -1 or (npc[12] and self.get_l1(self.position, (npc[5], npc[6])) > 5) \
                        or self.get_l1(self.position, (npc[5], npc[6])) > 6 or \
                        (self.win_rate(tmp_npc)>1 and self.get_l1(self.position, (npc[5],npc[6]))>=5):
                    self.attack_pos = (npc[5],npc[6])
                    break
                return (0, 0)
        if self.role == 'attacker':
            min_dis = 14
        else :
            min_dis=4
        for enemy in self.enemies:
            if self.win_rate(enemy)>0.7:
                enemy_pos = (enemy[5], enemy[6])
                dis = self.get_attack_dis(self.position, enemy_pos)
                if dis <= 0:
                    return (0, 0)
                if dis <= min_dis:
                    min_dis = dis
                    choose_enemy = enemy_pos
        if choose_enemy:
            dis, path = self.dijkstra('des', choose_enemy)
            direction = (path[0][0] - self.position[0], path[0][1] - self.position[1])
            return direction

        choose_npc=None
        min_dis=6
        if self.role=='hider' or len(self.kill_npc)>=1:
            min_dis=2

        self.NPCs.sort(key=lambda x:x[4])
        for npc in self.NPCs:
            if int(npc[4])==-1 or self.win_rate(npc)>1 or self.get_l1(self.position,(npc[5],npc[6]))>5:
                npc_pos=(npc[5],npc[6])
                dis=self.get_linf(self.position,npc_pos)
                if dis<=4:
                    tmp_npc=copy.deepcopy(npc)
                    tmp_npc[5] = self.position[0] + self.get_l1(self.position,npc_pos) - 1
                    tmp_npc[6] = self.position[1]
                    if int(npc[4])==-1 or (npc[12] and self.get_l1(self.position,(npc[5],npc[6]))>5) \
                            or self.get_l1(self.position,(npc[5],npc[6]))>6 or \
                            (self.win_rate(tmp_npc)>1 and self.get_l1(self.position, (npc[5],npc[6]))>=5):
                        self.attack_pos=npc_pos
                        return None
                    return (0,0)
                if self.get_attack_dis(self.position,npc_pos)<=min_dis:
                    min_dis=dis
                    choose_npc=npc_pos

        if choose_npc:
            dis, path = self.dijkstra('des', choose_npc)
            direction=(path[0][0]-self.position[0],path[0][1]-self.position[1])
        return direction

    def dijkstra(self,end_type,end_position):
        foods_tmp=copy.deepcopy(self.forest)
        OPEN=[]
        PARENT=dict()
        g=dict()
        danger={}
        all_obses=set(self.obstacle +self.obstacle_entity + self.obstacle_foods)

        for enemy in self.enemies:
            if self.win_rate(enemy)>0.5:
                continue
            for i in range(-4,5):
                for j in range(-4,5):
                    if (enemy[5]+i,enemy[6]+j) in all_obses:
                        continue
                    threat=10/max(min(abs(i),abs(j)),1)
                    enemy_pos=(enemy[5]+i,enemy[6]+j)
                    if enemy_pos in danger.keys():
                        danger[enemy_pos] += threat
                    else:
                        danger[enemy_pos] = threat
        if self.attacking and self.attacking_info[4] in [-2,-3] and self.win_rate(self.attacking_info)<0.6:
            for i in range(-4,5):
                for j in range(-4,5):
                    enemy_pos=(self.attacking_info[5],self.attacking_info[6])
                    danger_pos=(self.attacking_info[5]+i,self.attacking_info[6]+j)
                    if danger_pos in all_obses or self.get_l1(danger_pos,enemy_pos)>4 or self.get_l1(danger_pos,enemy_pos)<1:
                        continue
                    threat=10/max(1,self.get_l1(danger_pos,enemy_pos))
                    if danger_pos in danger.keys():
                        danger[danger_pos] += threat
                    else:
                        danger[danger_pos] = threat

        PARENT[self.position]=[[self.position,0]]
        g[self.position]=(0,self.food_carried,self.water_carried)
        if end_type=='end':
            heapq.heappush(OPEN, (self.get_linf(self.position,end_position), self.position))
        if end_type=='des':
            heapq.heappush(OPEN, (self.get_linf(self.position,end_position), self.position))
        elif end_type == 'evade_npc':
            heapq.heappush(OPEN, (6 - self.get_l1(self.position, end_position)+ self.get_linf(self.position, end_position), self.position))
        elif end_type=='r':
            heapq.heappush(OPEN, (abs(self.position[0]-end_position), self.position))
        elif end_type=='c':
            heapq.heappush(OPEN, (abs(self.position[1]-end_position), self.position))
        else:
            heapq.heappush(OPEN, (0, self.position))

        # if self.end_type ==end_type:
        #     end_position=self.end
        end=None
        cutoff=150
        closestPos=None
        while OPEN:
            _,s =heapq.heappop(OPEN)
            if (end_type=='water' and s in self.waters) or (end_type=='food' and s in self.forest):
                end=s
                break
            elif (end_type=='r' and s[0]==end_position) or (end_type=='c' and s[1]==end_position):
                end=s
                break
            elif (end_type =='end' and self.get_linf(s,end_position)<4):
                end=s
                break
            elif (end_type =='des' and self.get_linf(s,end_position)==1):
                end=s
                break
            elif (end_type=='evade_npc' and self.get_l1(s,end_position)>6):
                end=s
                break
            cutoff -= 1
            if cutoff <= 0:
                end = closestPos
                break

            for s_n in self.get_neighbor(s):
                if s_n in all_obses:
                    continue
                new_cost=self.dijk_cost(g[s],s_n,danger,foods_tmp)
                if s_n not in g.keys():
                    g[s_n]=(math.inf,0,0)
                if new_cost[0]<g[s_n][0]:
                    g[s_n]=new_cost
                    closestPos=s_n
                    PARENT[s_n]=[s,new_cost[0]]

                    if end_type == 'end':
                        heapq.heappush(OPEN, (new_cost[0]+self.get_attack_dis(s_n, end_position), s_n))
                    if end_type == 'des':
                        heapq.heappush(OPEN, (new_cost[0]+self.get_attack_dis(s_n, end_position), s_n))
                    elif end_type == 'r':
                        heapq.heappush(OPEN, (new_cost[0]+1.5*abs(s_n[0] - end_position), s_n))
                    elif end_type == 'c':
                        heapq.heappush(OPEN, (new_cost[0]+1.5*abs(s_n[1] - end_position), s_n))
                    elif end_type =='evade_npc':
                        heapq.heappush(OPEN,(new_cost[0]+1.5*(8-self.get_l1(s_n, end_position)+ self.get_linf(s_n, end_position)), s_n))
                    else:
                        heapq.heappush(OPEN,(new_cost[0],s_n))
        # if cutoff <= 0:
        #     self.end = end_position
        #     self.end_type = end_type
        # else:
        #     self.end = None
        #     self.end_type = None
        try:
            return self.extract_path(self.position,end,PARENT)
        except:
            for neighbor in self.get_neighbor(self.position):
                if neighbor not in all_obses:
                    return 0,[neighbor]
            else:
                return 0,[self.position]

    def dijk_cost(self,gs,end,danger,foods_tmp):

        cost=gs[0]+1
        food=gs[1]-1
        water=gs[2]-1

        if end in danger.keys():
            cost+=danger[end]
        if end in foods_tmp:
            foods_tmp.remove(end)
            food =self.food_max
        else:
            cost_tmp= (1-gs[1]/self.food_max)*2
            if gs[1]<5:
                cost_tmp*=3
            cost+=cost_tmp
        if end in self.waters:
            water=self.water_max
        else:
            cost_tmp= (1 - gs[2] / self.water_max)
            if gs[2]<5:
                cost_tmp *= 3
            cost += cost_tmp

        return (cost,food,water)

    def extract_path(self,start,end,PARENT):
        path = [end]
        s=end
        while True:
            s1=PARENT[s]
            s=s1[0]
            if s ==start:
                break
            path.append(s)
        path.reverse()
        return len(path), path

    def get_near_des(self,destination):
        neighbor=[(0,0),(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1),(-2,-2),(-2,-1),(-2,0),(-2,1),(-2,2),
                  (-1,-2),(-1,2),(0,-2),(0,2),(1,-2),(1,2),(2,-2),(2,-1),(2,0),(2,1),(2,2)]
        for neighbor_pos in neighbor:
            cur_pos=(neighbor_pos[0]+destination[0],neighbor_pos[1]+destination[1])
            if cur_pos in self.obstacle or cur_pos in self.obstacle_entity or cur_pos in self.obstacle_foods:
                continue
            return cur_pos
        return destination
    def get_neighbor(self,s):
        motions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # todo check env
        neighbor=[(s[0] + u[0], s[1] + u[1]) for u in motions]
        random.shuffle(neighbor)
        return neighbor
    def get_linf(self,start,goal):
        sr, sc = start
        gr, gc = goal
        return max(abs(gr - sr), abs(gc - sc))
    def get_l1(self,start,goal):
        sr, sc = start
        gr, gc = goal
        return abs(gr - sr)+abs(gc - sc)

    def get_attack_dis(self,start,goal):
        dis_l1 = self.get_l1(start, goal)
        dis_linf = self.get_linf(start,goal)
        if dis_l1 - dis_linf > 4:
            dis = max(0,dis_l1 - 8)
        else:
            dis = max(0, dis_linf - 4)
        return dis
    def get_eucl(self,start,goal):
        sr, sc = start
        gr, gc = goal
        return math.sqrt((gr - sr)**2+(gc - sc)**2)
    def get_best_dir(self,goal,obses):
        dic={}
        new_pos=self.position
        for neighbor in self.get_neighbor(self.position):
            if neighbor in obses:
                continue
            if neighbor in self.searched.keys():
                dic[neighbor]=self.searched[neighbor]
            else:
                dic[neighbor]=self.get_linf(self.position,goal)+self.get_linf(self.position,goal)
        for key,value in dic.items():
            if value==min(dic.values()):
                dic.pop(key)
                new_pos=key
                break
        if dic:
            for key, value in dic.items():
                if value == min(dic.values()):
                    if self.position in self.searched.keys():
                        self.searched[self.position]+=2
                    else:
                        self.searched[self.position]=value
                    break
        return new_pos
    def get_danger_dir(self):
        rep=(0.1,0.1)
        for enemy in self.enemies:
            if self.win_rate(enemy)>0.4 or self.get_attack_dis((enemy[5],enemy[6]),self.position)>=6:
                continue
            rep=(rep[0]+self.position[0]-enemy[5],rep[1]+self.position[1]-enemy[6])
        if rep==(0.1,0.1):
            return None
        dir=(int(7*rep[0]/math.sqrt(rep[0]**2+rep[1]**2)),int(7*rep[1]/math.sqrt(rep[0]**2+rep[1]**2)))
        new_pos=(min(144,max(16,self.position[0]+dir[0])),min(144,max(16,self.position[1]+dir[1])))
        new_pos=self.get_near_des(new_pos)
        return new_pos
    def win_rate(self,enemy):
        #combat 还需要除以4，constitution除以2
        exp_list=[6, 13, 21, 30, 40, 51, 63, 76, 91, 107, 125, 145, 167, 191, 218, 248, 281, 317, 357, 402, 451, 505, 565,
         631, 704, 785, 874, 973, 1082, 1202, 1335, 1482, 1644, 1823, 2020, 2238, 2478, 2743, 3036, 3359, 3716, 4110,
         4545, 5026, 5557, 6143, 6790, 7504, 8292, 9162]

        if enemy[4]<0:
            enemy_def_level=enemy[3]*2
            enemy_skill_level=enemy[3]
            enemy_dis=self.get_l1(self.position,(enemy[5],enemy[6]))
        else:
            enemy_def_level=bisect.bisect_left(exp_list,enemy[7]*4)
            enemy_skill_level=20
            enemy_dis = self.get_linf(self.position, (enemy[5], enemy[6]))
            for i in range(50):
                hp=bisect.bisect_left(exp_list,exp_list[i]/2+exp_list[9])
                if 0.25*hp+0.5*i>enemy[3]-0.25*enemy_def_level:
                    enemy_skill_level = i - 1+enemy[3]*0.1 #equipment??
                    break

        my_skill_level=20
        my_def_level=bisect.bisect_left(exp_list,self.info[7]*4)
        for i in range(50):
            hp=bisect.bisect_left(exp_list,exp_list[i]/2+exp_list[9])
            if 0.25*hp+0.5*i>self.info[3]-0.25*my_def_level:
                my_skill_level = i - 1+self.info[3]*0.1
                break

        if enemy_dis==1:
            enemy_damage_base= np.floor(7 + enemy_skill_level * 63 / 99)
        elif enemy_dis<=3:
            enemy_damage_base=  np.floor(3 + enemy_skill_level * 32 / 99)
        else:
            enemy_damage_base= np.floor(1 + enemy_skill_level * 24 / 99)
        my_damage_base =np.floor(1 + my_skill_level * 24 / 99)

        my_damage=max(1,(10+my_skill_level-0.3*enemy_def_level-0.7*enemy_skill_level)/20*my_damage_base+ \
                     (10 - my_skill_level + 0.3 * enemy_def_level + 0.7 * enemy_skill_level) / 20)
        if enemy_dis<=3:
            my_skill_level=1 #一般没有练前两个技能
        enemy_damage=max(1,(10+enemy_skill_level-0.3*my_def_level-0.7*my_skill_level)/20*enemy_damage_base+ \
                     (10 - enemy_skill_level + 0.3 * my_def_level + 0.7 * my_skill_level) / 20)

        win=(5+self.health/enemy_damage-1.1*enemy[11]/my_damage)/10
        return win


class Submission:
    team_klass = MyTeam
    init_params = {}
