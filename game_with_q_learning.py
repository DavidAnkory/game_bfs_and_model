import pygame
import heapq
import numpy as np
import random
from collections import deque
import time

# אתחול המשחק
pygame.init()
WIDTH, HEIGHT, GRID_SIZE = 500, 500, 25
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# הגדרת השחקן והרודף
player = pygame.Rect(50, 50, GRID_SIZE, GRID_SIZE)
chaser = pygame.Rect(400, 400, GRID_SIZE, GRID_SIZE)
speed = GRID_SIZE

dash_speed = GRID_SIZE * 2  # דש מאפשר לשחקן לנוע מהר יותר לזמן קצר
cooldown = 0  # משתנה להגבלת תדירות הדש

# יצירת מכשולים במפה
obstacles = [(100, 100), (200, 200), (300, 300), (150, 250)]


# פונקציות עזר
def is_valid_move(x, y):
    return 0 <= x < WIDTH and 0 <= y < HEIGHT and (x, y) not in obstacles


# פונקציית A* למציאת הנתיב היעיל ביותר
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_path(start, goal):
    queue = []
    heapq.heappush(queue, (0, start, []))
    visited = set()
    while queue:
        cost, (x, y), path = heapq.heappop(queue)
        if (x, y) == goal:
            return path
        for dx, dy in [(-speed, 0), (speed, 0), (0, -speed), (0, speed)]:
            next_pos = (x + dx, y + dy)
            if is_valid_move(*next_pos) and next_pos not in visited:
                visited.add(next_pos)
                new_cost = cost + 1 + heuristic(next_pos, goal)
                heapq.heappush(queue, (new_cost, next_pos, path + [next_pos]))
    return []


# למידת חיזוקים (Q-learning)
Q_table = np.zeros((WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE, 4))  # טבלה לשמירת Q-values
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.2  # סיכוי לחקור פעולה אקראית

actions = [(-speed, 0), (speed, 0), (0, -speed), (0, speed)]  # פעולות אפשריות (מעלה, למטה, ימין, שמאל)


def choose_action(chaser_pos):
    if np.random.rand() < epsilon:
        return random.choice(actions)  # חקור פעולה אקראית
    x, y = chaser_pos[0] // GRID_SIZE, chaser_pos[1] // GRID_SIZE
    return actions[np.argmax(Q_table[x, y])]  # בחר פעולה עם הערך הכי גבוה ב-Q-table


def update_q_table(chaser_pos, action, reward, new_pos):
    x, y = chaser_pos[0] // GRID_SIZE, chaser_pos[1] // GRID_SIZE
    nx, ny = new_pos[0] // GRID_SIZE, new_pos[1] // GRID_SIZE
    action_index = actions.index(action)
    best_next = np.max(Q_table[nx, ny])
    Q_table[x, y, action_index] += learning_rate * (reward + discount_factor * best_next - Q_table[x, y, action_index])


# אתחול משתנים
running = True
start_time = time.time()  # זמן התחלת המשחק

while running:
    screen.fill((30, 30, 30))
    pygame.draw.rect(screen, (0, 255, 0), player)
    pygame.draw.rect(screen, (255, 0, 0), chaser)
    for obs in obstacles:
        pygame.draw.rect(screen, (100, 100, 100), (obs[0], obs[1], GRID_SIZE, GRID_SIZE))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # תנועת השחקן
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and is_valid_move(player.x - speed, player.y): player.x -= speed
    if keys[pygame.K_RIGHT] and is_valid_move(player.x + speed, player.y): player.x += speed
    if keys[pygame.K_UP] and is_valid_move(player.x, player.y - speed): player.y -= speed
    if keys[pygame.K_DOWN] and is_valid_move(player.x, player.y + speed): player.y += speed

    # דש עם מגבלת זמן
    if keys[pygame.K_SPACE] and cooldown == 0:
        player.x += dash_speed if keys[pygame.K_RIGHT] else -dash_speed if keys[pygame.K_LEFT] else 0
        player.y += dash_speed if keys[pygame.K_DOWN] else -dash_speed if keys[pygame.K_UP] else 0
        cooldown = 10
    if cooldown > 0:
        cooldown -= 1

    # בחירת שיטת הניווט
    path = astar_path((chaser.x, chaser.y), (player.x, player.y))  # אפשר להחליף ב-bfs_path או Q-learning
    if path:
        chaser.x, chaser.y = path[0]

    # חישוב תגמול
    reward = 0
    if chaser.colliderect(player):
        reward = 100  # תגמול חיובי אם הרודף תפס את השחקן
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        game_over_message = f"המשחק נגמר! הזמן שחלף: {elapsed_time} שניות."
        print(game_over_message)
        running = False
    elif (player.x, player.y) in obstacles:
        reward = 50  # תגמול חיובי אם השחקן נוגע במכשול
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        game_over_message = f"המשחק נגמר! הזמן שחלף: {elapsed_time} שניות."
        print(game_over_message)
        running = False

    # עדכון ה-Q-table של הרודף
    if running:
        action = choose_action((chaser.x, chaser.y))
        update_q_table((chaser.x, chaser.y), action, reward, (chaser.x + action[0], chaser.y + action[1]))

    # הצגת הודעה על המסך כאשר המשחק נגמר
    if not running:
        font = pygame.font.Font(None, 36)
        text = font.render(game_over_message, True, (255, 0, 0))
        screen.blit(text, (WIDTH // 4, HEIGHT // 2))

    pygame.display.flip()
    clock.tick(10)

pygame.quit()
