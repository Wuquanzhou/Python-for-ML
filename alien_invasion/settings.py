class Settings():
    def __init__(self):
        #屏幕属性
        self.screen_width = 1200
        self.screen_height = 800
        self.bg_color = (230,230,230)

        #飞船每次移动像素
        self.ship_limit = 3

        #外星人每次移动像素

        self.fleet_drop_speed = 10

        self.speedup_scale = 1.1

        self.score_scale = 1.5

        self.initialize_dynamic_settings()

        #子弹属性

        self.bullet_width = 3
        self.bullet_height = 15
        self.bullet_color = 60, 60,60
        self.bullets_allowed = 3

    def initialize_dynamic_settings(self):
        self.ship_speed_factor = 1.5
        self.bullet_speed_factor = 3
        self.alien_speed_factor = 1
        #1表示向右，-1表示向左
        self.fleet_direction = 1

        #记分
        self.alien_points = 50

    def increase_speed(self):
        self.ship_speed_factor *= self.speedup_scale
        self.bullet_speed_factor *= self.speedup_scale
        self.alien_speed_factor *= self.speedup_scale

        self.alien_points = int(self.alien_points * self.score_scale)

        print(self.alien_points)