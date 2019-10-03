import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.uix.image import Image
import time
from NeuroPy2 import NeuroPy
from time import sleep
import pickle
from collections import deque
import numpy as np
from scipy.signal import find_peaks
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.properties import (
    NumericProperty, ReferenceListProperty, ObjectProperty
)
from kivy.vector import Vector
from kivy.clock import Clock
import random
from kivy.core.window import Window

from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)

dataNameList = ['attention','meditation','rawValue','delta','theta','lowAlpha','highAlpha',
            'lowBeta','highBeta','lowGamma','midGamma','poorSignal']

dataNameList = ['attention','meditation','rawValue','delta','theta','lowAlpha','highAlpha',
            'lowBeta','highBeta','lowGamma','midGamma','poorSignal']
featureList = ['attention','meditation','rawValue','delta','theta','lowAlpha','highAlpha',
            'lowBeta','highBeta','lowGamma','midGamma']

labels = ['focus','relax', 'upWord', 'downWord', 
          'upColor', 'downColor', 
          'CyanUP','greenDOWN', 'yellowRIGHT', 'BlackLEFT']#,'blink']

labels = ['relax','upColor','CyanUP']

n_label = len(labels)

trainDataDict = dict()
for data in dataNameList:
    trainDataDict[data] = []

def load_data(dataDict, label, count):    
    for data in dataNameList:
        dataDict[data].append(np.load('model/dataset/{}/{}/{}.npy'.format(label,count,data))[:100])


n_samples = 30
test_n_samples = int(n_samples/2)
test_size = n_label * int(n_samples/2)
train_n_samples = round(n_samples/2)
train_size = n_label * round(n_samples/2)
#nums = np.arange(n_samples)*2
nums = np.arange(n_samples)
trainNums = np.concatenate([nums[:5],nums[10:15],nums[20:25]])#,nums[31:41], nums[51:61],nums[71:81]])
#trainNums = nums[:5]
np.random.shuffle(trainNums)

for label in labels:
    for i in trainNums:
        load_data(trainDataDict,label, i)
        

for data in dataNameList:
    trainDataDict[data] = np.array(trainDataDict[data])

#connect features
trainData = []
for data in featureList:
    trainData.append(trainDataDict[data])
trainData = np.array(trainData).transpose(1,0,2)

trainLabels = []
for i in range(n_label):
    trainLabels.append(np.ones(int(n_samples/2))*i )#,np.ones(15)*2])
trainLabels = np.concatenate(trainLabels)
train_indexes = np.arange(len(trainLabels))
np.random.shuffle(train_indexes)



x_train = trainData[train_indexes]

img_rows, img_cols = 10, 10
channel = 11

x_train = x_train.astype('float32')
scaler = MinMaxScaler()
print(scaler.fit(x_train.reshape(-1, 1100)))


GLOBAL_TIMER_VALUE = 1      # in seconds
COLOR_CHOOSEN = (1,1,1,0.7)
COLOR_OTHERS = (1,1,1,1)
COLOR_TRANSPARENT = (0,0,0,0)

PREDICTED_Y = 2
PREDICTED_PROBA = 0.0


#--------------------


def load_model():
    loaded_model = pickle.load(open('knn_best.pkl', 'rb'))
    return loaded_model

model = load_model()
#preds = np.array(loaded_model.predict(testData.reshape(l_n_samples, -1)))

def init_DataDict():
    for data in dataNameList:
        data_dict[data] = deque(maxlen=1000)


neuropy = NeuroPy("/dev/cu.MindWaveMobile-SerialPo-8",115200)
neuropy.start()
#python3 -m serial.tools.list_ports

data_dict = dict()

init_DataDict()


def find_peak(raw_values):
    mean = np.mean(raw_values)
    peaks, _ = find_peaks(np.abs(raw_values), height=mean * 4, threshold=None, distance=15)
    if len(peaks) > 1:
        print("\nSINGLE CLICK")
        #print(len(peaks))
        return 1
    else:
        return 0

def predict(model,values):
    values = scaler.transform(values.reshape(-1, 1100))
    preds = int(np.array(model.predict(values))[0])
    print('\npreds  : ', preds)
    return preds

def read_data():
    print("\r data:  {}   signal: {}".format(getattr(neuropy,'rawValue'),getattr(neuropy,'poorSignal')), end='')
    for data in dataNameList:
        data_dict[data].append(getattr(neuropy,data))


GLOBAL_TIMER_VALUE = 0.01      # in seconds
COLOR_CHOOSEN = (1,1,1,0.7)
COLOR_OTHERS = (1,1,1,1)
COLOR_TRANSPARENT = (0,0,0,0)


#-------------------------------------------------

class PongPaddle(Widget):
    score = NumericProperty(0)

    def bounce_ball(self, ball):
        if self.collide_widget(ball):
            vx, vy = ball.velocity
            offset = (ball.center_y - self.center_y) / (self.height / 2)
            bounced = Vector(-1 * vx, vy)
            vel = bounced 
            ball.velocity = vel.x, vel.y + offset


class PongBall(Widget):
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self):
        self.pos = Vector(*self.velocity) + self.pos


class PongGame(Widget):
    ball = ObjectProperty(None)
    player1 = ObjectProperty(None)
    player2 = ObjectProperty(None)
    count = 1
    pred_y = 2

    def serve_ball(self, vel=(6, 0)):
        self.ball.center = self.center
        self.ball.velocity = vel


    def update(self, dt):
        read_data()
        if len(data_dict['rawValue']) > 100:
            blink = find_peak(np.array(data_dict['rawValue'])[:100])

            testData = []
            for data in featureList:
                testData.append(np.array(data_dict[data])[:100])
            testData = np.array(testData)
            self.pred_y = int(predict(model,testData))
    

        if(self.pred_y == 0):
            self.player1.center_y += 5
        elif(self.pred_y == 1):
            self.player1.center_y -=5
        else:
            #topla farkı indiriyor jump
            self.player1.center_y -= (self.player1.center_y - self.ball.center_y)

        self.ball.move()

        # bounce of paddles
        self.player1.bounce_ball(self.ball)
        self.player2.bounce_ball(self.ball)

        # bounce ball off bottom or top
        if (self.ball.y < self.y) or (self.ball.top > self.top):
            #print('ball: ', self.ball.y)
            self.ball.velocity_y *= -1

        # Window size 1600,1200
        if(self.player1.y > 1050 ):
            self.player1.y = 1050
        if(self.player1.y < 100):
            self.player1.y = 100

        
        if(self.player2.y > 1050 ):
            self.player2.y = 1050
        if(self.player2.y < 100):
            self.player2.y = 100
        

        # went of to a side to score point?
        if self.ball.x < self.x:
            self.player2.score += 1
            self.serve_ball(vel=(4, 0))
        if self.ball.x > self.width:
            self.player1.score += 1
            self.serve_ball(vel=(-4, 0))

        #random.randint(-15,15)
        
        if self.count >= 10:
            self.player2.center_y -= (self.player2.center_y - self.ball.center_y)/10 + random.randint(-4,4)
            self.count = 0
        self.count += 1


class PongPage2():

    class PongApp(App):
        def build(self):
            #Window.clearcolor = (0, 0, 1, 0)
            game = PongGame()
            game.serve_ball()
            
            Clock.schedule_interval(game.update, 1.0 / 100.0)

            return game

#--------------------   

class MainPage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.cols = 1
        self.rows = 5
        self.button_rows = 5
        
        self.current_option = 0
        self.current_screen = False
        
        self.stupid_counter = 0
        
        self.button1 = Button(text="Kişiler",background_normal="buttons/contact_button.png", font_size=35)
        self.button1.bind(on_press=self.contacts_button)
        self.add_widget(self.button1)
        
        self.button2 = Button(text="Kitaplar",background_normal="buttons/book_button.png", font_size=35)
        self.button2.bind(on_press=self.books_button)
        self.add_widget(self.button2)
        
        self.button3 = Button(text="Pong Oyunu",background_normal="buttons/pong_button.png", font_size=35)
        self.button3.bind(on_press=self.pong_button)
        self.add_widget(self.button3)
        
        self.button4 = Button(text="Klavye",background_normal="buttons/keys_button.png", font_size=35)
        self.button4.bind(on_press=self.keyboard_button)
        self.add_widget(self.button4)
        
        self.button5 = Button(text="Çıkış",background_normal="buttons/back_button.png", font_size=35)
        self.button5.bind(on_press=self.quit_button)
        self.add_widget(self.button5)
        
        Clock.schedule_interval(lambda dt: self.callback_f(), GLOBAL_TIMER_VALUE)
        
        self.update_texts()
        
        

    def contacts_button(self, instances):
        self.set_current_screen(False)
        main_app.screen_manager.current = "Contacts"
        main_app.contacts_page.set_current_screen(True)
    
    def books_button(self, instances):
        self.set_current_screen(False)
        main_app.screen_manager.current = "Books"
        main_app.books_page.set_current_screen(True)
    
    def pong_button(self, instances):
        self.set_current_screen(False)
        main_app.screen_manager.current = "Pong"
        main_app.pong_page.set_current_screen(True)

    def keyboard_button(self, instances):
        self.set_current_screen(False)
        main_app.screen_manager.current = "Keyboard"
        main_app.keyboard_page.set_current_screen(True)
    
    def quit_button(self, instances):
        App.get_running_app().stop()



    def next_option(self):
        if(self.current_option < self.button_rows-1):
            self.current_option += 1
            self.update_texts()
            
    def previous_option(self):
        if(self.current_option > 0):
            self.current_option -= 1
            self.update_texts()
    
    def update_texts(self):
        if(self.current_option == 0):
            self.button1.background_color = COLOR_CHOOSEN
        else:
            self.button1.background_color = COLOR_OTHERS
            
        if(self.current_option == 1):
            self.button2.background_color = COLOR_CHOOSEN
        else:
            self.button2.background_color = COLOR_OTHERS
            
        if(self.current_option == 2):
            self.button3.background_color = COLOR_CHOOSEN
        else:
            self.button3.background_color = COLOR_OTHERS
        if(self.current_option == 3):
            self.button4.background_color = COLOR_CHOOSEN
        else:
            self.button4.background_color = COLOR_OTHERS
        if(self.current_option == 4):
            self.button5.background_color = COLOR_CHOOSEN
        else:
            self.button5.background_color = COLOR_OTHERS
    
    def choose_current_option(self):
        if(self.current_option == 0):
            self.contacts_button(1)
        elif(self.current_option == 1):
            self.books_button(1)
        elif(self.current_option == 2):
            self.pong_button(1)
        elif(self.current_option == 3):
            self.keyboard_button(1)
        elif(self.current_option == 4):
            self.quit_button(1)
        else:
            print("ERROR - current_option = ", self.current_option)
    
    def set_current_screen(self, status):
        self.current_screen = status
    
    def callback_f(self):
        if(self.current_screen):
            #print(main_app.screen_manager.current, "Callback f is calling in every ", GLOBAL_TIMER_VALUE, " seconds.")
            read_data()
            
            if len(data_dict['rawValue']) > 100:
                blink = find_peak(np.array(data_dict['rawValue'])[:100])
                if blink == 1:
                    self.choose_current_option()
                testData = []
                for data in featureList:
                    testData.append(np.array(data_dict[data])[:100])
                testData = np.array(testData)
                predict(model,testData)
                
                init_DataDict()
                #    for _ in range(100):
                #       data_dict[data].popleft()
                if(self.current_option == self.button_rows-1):
                    self.current_option = -1
                self.next_option()

            #self.stupid_counter += 1
            #if(self.stupid_counter % 10 == 0):
            #    self.next_option()
            #print(self.stupid_counter)
            


class PongPage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.current_screen = False
        self.first_call_status = False

        self.pong=PongPage2().PongApp()



        self.button1 = Button(text="",background_normal="bg.png",size=(1600,1200))
        #self.button1.bind(on_press=self.first_call_f)
        self.add_widget(self.button1)
        
        Clock.schedule_interval(lambda dt: self.callback_f(), GLOBAL_TIMER_VALUE)
        
    
    
    def set_current_screen(self, status):
        self.current_screen = status
        
        
    def first_call_f(self):
        if not self.first_call_status:
            self.first_call_status = True
            
            icerik = self.pong.run()
            popup = Popup(title='Pong',
                          content=icerik,
                          size_hint=(None, None), size=(600, 800))

            popup.open()

            
        
    
    def callback_f(self):
        if(self.current_screen):
            if not self.first_call_status:
                self.first_call_f()
                self.first_call_status = True
                #print("----------")



class BooksPage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.cols = 1
        self.rows = 3
        self.button_rows = 3
        
        self.current_option = 0
        self.current_screen = False
        
        self.button_text_1 = "Beyin-Bilgisayar Arayüzü ve Kısa Tarihi"
        self.button_text_2 = "Biz Kimiz, Ne Yaptık?"

        self.button1 = Button(text=self.button_text_1,background_normal="buttons/e_button.png", font_size=35)
        self.button1.bind(on_press=self.book_1_button)
        self.add_widget(self.button1)
        
        self.button2 = Button(text=self.button_text_2,background_normal="buttons/e_button.png", font_size=35)
        self.button2.bind(on_press=self.book_2_button)
        self.add_widget(self.button2)
        
        self.button3 = Button(text="Ana Menü",background_normal="buttons/home_button.png", font_size=35)
        self.button3.bind(on_press=self.main_menu_button)
        self.add_widget(self.button3)
        
        Clock.schedule_interval(lambda dt: self.callback_f(), GLOBAL_TIMER_VALUE)
        
        self.update_texts()
        
        
    def book_1_button(self, instances):
        self.set_current_screen(False)
        main_app.screen_manager.current = "Book1"
        main_app.book_1_page.set_current_screen(True)
    
    def book_2_button(self, instances):
        self.set_current_screen(False)
        main_app.screen_manager.current = "Book2"
        main_app.book_2_page.set_current_screen(True)

    def main_menu_button(self, instances):
        self.set_current_screen(False)
        main_app.screen_manager.current = "Main"
        main_app.main_page.set_current_screen(True)
    
    
    def next_option(self):
        if(self.current_option < self.button_rows-1):
            self.current_option += 1
            self.update_texts()
            
    def previous_option(self):
        if(self.current_option > 0):
            self.current_option -= 1
            self.update_texts()
    
    def update_texts(self):
        if(self.current_option == 0):
            self.button1.background_color = COLOR_CHOOSEN
        else:
            self.button1.background_color = COLOR_OTHERS
            
        if(self.current_option == 1):
            self.button2.background_color = COLOR_CHOOSEN
        else:
            self.button2.background_color = COLOR_OTHERS
            
        if(self.current_option == 2):
            self.button3.background_color = COLOR_CHOOSEN
        else:
            self.button3.background_color = COLOR_OTHERS

    
    def choose_current_option(self):
        if(self.current_option == 0):
            self.book_1_button(1)
        elif(self.current_option == 1):
            self.book_2_button(1)
        elif(self.current_option == 2):
            self.main_menu_button(1)
        else:
            print("ERROR - current_option = ", self.current_option)
    
    def set_current_screen(self, status):
        self.current_screen = status
    
    def callback_f(self):
        if(self.current_screen):
            read_data()
            if len(data_dict['rawValue']) > 100:
                blink = find_peak(np.array(data_dict['rawValue'])[:100])
                if blink == 1:
                    self.choose_current_option()
                testData = []
                for data in featureList:
                    testData.append(np.array(data_dict[data])[:100])
                testData = np.array(testData)
                #predict(model,testData)
                
                init_DataDict()
                #    for _ in range(100):
                #       data_dict[data].popleft()
                if(self.current_option == self.button_rows-1):
                    self.current_option = -1
                self.next_option()
            

class Book1Page(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.cols = 1
        self.rows = 1
        self.button_rows = 1
        
        self.current_option = 0
        self.current_screen = False
        
        self.total_pages = 2
        self.page1 = True
        
        self.button_text_1 = "Beyin-bilgisayar arayüzü (BBA), beyinle elektronik \ncihazlar arasında doğrudan veri alışverişi sağlayan sistemlerdir. \nBu yolla harici bir cihaza ileri yönlü veri gönderilebilmekte ve cihazlardan veri \nalınabilmektedir. BBA teknolojisinin başlıca kullanım alanı medikal ve \naskeri alanlarda olmasına karşın bu kapsam zamanla genişlemektedir.\n\nİnsanlar üzerinde yapılan ilk BBA araştırmaları 1960'larda yapıldı. \nÇalışmada deneklerin, beyin dalgalarını ölçen elektroensefalografi (EEG) \nmetodu ile bir slayt göstericisini kontrol etmeleri sağlandı. Bu çalışma her \nne kadar bütün bir araştırma odağının başlangıcı olsa da henüz BBA ismini \nalmamıştı. “Beyin-bilgisayar arayüzü” deyişinin asıl \ndoğuşu 1970'lere denk gelmekte."
        self.button_text_2 = "Günümüzde BBA teknolojisinin geldiği noktada:\n\n * Uzvunu kaybetmiş insanlar için çok fonksiyonlu prostetik uzuvlar \ngeliştirilebiliyor.\n * Locked-in Syndrome olarak da isimlendirilen, kişiyi dünya ile iletişim \n  kurmayı neredeyse imkansız hale getirebilen ALS gibi hastalıklara \nsahip insanlara iletişim imkanı sunabiliyor.\n * İnsanlara meditasyon, öğrenme, hatırlama gibi bilişsel işlevler \nhakkında destek olunabiliyor.\n * İnsanların çeşitli harici elektronik cihazları kontrol etmeleri \nsağlanabiliyor.\n * Duyu organı hasarları için insan yapımı cihazlar üretilebiliyor."

        self.button1 = Button(text=self.button_text_1, font_size=40)
        self.button1.bind(on_press=self.page_button)
        self.add_widget(self.button1)
        

        
        Clock.schedule_interval(lambda dt: self.callback_f(), GLOBAL_TIMER_VALUE)
        
        
        
    def page_button(self, instances):
        self.button1.text = self.button_text_2
        self.button1.bind(on_press=self.main_menu_button)
        self.current_option += 1
        self.page1 = False
    
    def main_menu_button(self, instances):
        self.set_current_screen(False)
        main_app.screen_manager.current = "Main"
        main_app.main_page.set_current_screen(True)
    
    
    def next_option(self):
        if(self.page1):
            self.page_button(1)
        else:
            self.main_menu_button(1)



    def choose_current_option(self):
        if(self.current_option == 0):
            self.page_button(1)
        elif(self.current_option == 1):
            self.main_menu_button(1)
        else:
            print("ERROR - current_option = ", self.current_option)
    
    def set_current_screen(self, status):
        self.current_screen = status
        if(status):
            self.page1 = True
            self.button1.text = self.button_text_1
            self.button1.bind(on_press=self.page_button)
            self.current_option = 0

    
    def callback_f(self):
        if(self.current_screen):
            read_data()
            if len(data_dict['rawValue']) > 100:
                blink = find_peak(np.array(data_dict['rawValue'])[:100])
                if blink == 1:
                    self.next_option()
                testData = []
                for data in featureList:
                    testData.append(np.array(data_dict[data])[:100])
                testData = np.array(testData)
                predict(model,testData)
                
                init_DataDict()
                #    for _ in range(100):
                #       data_dict[data].popleft()
                if(self.current_option == self.button_rows-1):
                    self.current_option = -1
                



class Book2Page(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.cols = 1
        self.rows = 1
        self.button_rows = 1
        
        self.current_option = 0
        self.current_screen = False
        
        self.total_pages = 2
        self.page1 = True
        
        self.button_text_1 = "Biz Kimiz\n\nAnıl Osman Tur\n\nSinan Gençoğlu\n\nMert Bacak"
        self.button_text_2 = "Ne Yaptık ...\nKullanım kolaylığı sağlayan bir arayüzle kişinin temel bilgisayar\nişlemlerini gerçekleştirmesini sağlayan bir paket yazılım."

        self.button1 = Button(text=self.button_text_1, font_size=40)
        self.button1.bind(on_press=self.page_button)
        self.add_widget(self.button1)
        

        
        Clock.schedule_interval(lambda dt: self.callback_f(), GLOBAL_TIMER_VALUE)
        
        
        
    def page_button(self, instances):
        self.button1.text = self.button_text_2
        self.button1.bind(on_press=self.main_menu_button)
        self.current_option += 1
        self.page1 = False
    
    def main_menu_button(self, instances):
        self.set_current_screen(False)
        main_app.screen_manager.current = "Main"
        main_app.main_page.set_current_screen(True)
    
    
    def next_option(self):
        if(self.page1):
            self.page_button(1)
        else:
            self.main_menu_button(1)



    def choose_current_option(self):
        if(self.current_option == 0):
            self.page_button(1)
        elif(self.current_option == 1):
            self.main_menu_button(1)
        else:
            print("ERROR - current_option = ", self.current_option)
    
    def set_current_screen(self, status):
        self.current_screen = status
    
    def callback_f(self):    
        if(self.current_screen):
            read_data()
            if len(data_dict['rawValue']) > 100:
                blink = find_peak(np.array(data_dict['rawValue'])[:100])
                if blink == 1:
                    self.next_option()
                testData = []
                for data in featureList:
                    testData.append(np.array(data_dict[data])[:100])
                testData = np.array(testData)
                predict(model,testData)
                
                init_DataDict()
                #    for _ in range(100):
                #       data_dict[data].popleft()
                if(self.current_option == self.button_rows-1):
                    self.current_option = -1


class ContactsPage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.cols = 1
        self.rows = 4
        self.button_rows = 4
        
        self.current_option = 0
        self.current_screen = False
        
        self.button_text_1 = "Anıl Osman Tur : +90 222 3333"
        self.button_text_2 = "Sinan Gençoğlu : +90 222 3333"
        self.button_text_3 = "Mert Bacak : +90 222 3333"

        self.button1 = Button(text=self.button_text_1,background_normal="buttons/e_button.png", font_size=35)
        self.button1.bind(on_press=self.action_button)
        self.add_widget(self.button1)
        
        self.button2 = Button(text=self.button_text_2,background_normal="buttons/e_button.png", font_size=35)
        self.button2.bind(on_press=self.action_button)
        self.add_widget(self.button2)
        
        self.button3 = Button(text=self.button_text_3,background_normal="buttons/e_button.png", font_size=35)
        self.button3.bind(on_press=self.action_button)
        self.add_widget(self.button3)
        
        self.button4 = Button(text="Ana Menü",background_normal="buttons/home_button.png", font_size=35)
        self.button4.bind(on_press=self.main_menu_button)
        self.add_widget(self.button4)
        
        Clock.schedule_interval(lambda dt: self.callback_f(), GLOBAL_TIMER_VALUE)
        
        self.update_texts()
        
    
    def action_button(self, instances):
        if(self.current_option == 0):
            info = self.button_text_1[:-15]
        elif(self.current_option == 1):
            info = self.button_text_2[:-15]
        elif(self.current_option == 2):
            info = self.button_text_3[:-15]
        else:
            print("ERROR - ", self.current_option)
            info = ""
        main_app.action_page.set_call_info(info)
        
        self.set_current_screen(False)
        main_app.screen_manager.current = "Action"
        main_app.action_page.set_current_screen(True)

    def main_menu_button(self, instances):
        self.set_current_screen(False)
        main_app.screen_manager.current = "Main"
        main_app.main_page.set_current_screen(True)
    
    
    def next_option(self):
        if(self.current_option < self.button_rows-1):
            self.current_option += 1
            self.update_texts()
            
    def previous_option(self):
        if(self.current_option > 0):
            self.current_option -= 1
            self.update_texts()
    
    def update_texts(self):
        if(self.current_option == 0):
            self.button1.background_color = COLOR_CHOOSEN
        else:
            self.button1.background_color = COLOR_OTHERS
            
        if(self.current_option == 1):
            self.button2.background_color = COLOR_CHOOSEN
        else:
            self.button2.background_color = COLOR_OTHERS
            
        if(self.current_option == 2):
            self.button3.background_color = COLOR_CHOOSEN
        else:
            self.button3.background_color = COLOR_OTHERS
        
        if(self.current_option == 3):
            self.button4.background_color = COLOR_CHOOSEN
        else:
            self.button4.background_color = COLOR_OTHERS
    
    def choose_current_option(self):
        if(self.current_option == 0):
            self.action_button(1)
        elif(self.current_option == 1):
            self.action_button(1)
        elif(self.current_option == 2):
            self.action_button(1)
        elif(self.current_option == 3):
            self.main_menu_button(1)
        else:
            print("ERROR - current_option = ", self.current_option)
    
    def set_current_screen(self, status):
        self.current_screen = status
    
    def callback_f(self):
        if(self.current_screen):
            read_data()
            if len(data_dict['rawValue']) > 100:
                blink = find_peak(np.array(data_dict['rawValue'])[:100])
                if blink == 1:
                    self.choose_current_option()
                testData = []
                for data in featureList:
                    testData.append(np.array(data_dict[data])[:100])
                testData = np.array(testData)
                predict(model,testData)
                
                init_DataDict()
                #    for _ in range(100):
                #       data_dict[data].popleft()
                if(self.current_option == self.button_rows-1):
                    self.current_option = -1
                self.next_option()
   
        
        
class ActionPage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.cols = 1
        self.rows = 4
        self.button_rows = 4
        
        self.current_option = 0
        self.current_screen = False
        
        self.call_info = ""
        
        self.button1 = Button(text="Ara",background_normal="buttons/e_button.png", font_size=35)
        self.button1.bind(on_press=self.call_button)
        self.add_widget(self.button1)
        
        self.button2 = Button(text="Mesaj Gönder",background_normal="buttons/e_button.png", font_size=35)
        self.button2.bind(on_press=self.message_button)
        self.add_widget(self.button2)
        
        self.button3 = Button(text="Geri Dön",background_normal="buttons/back_button.png", font_size=35)
        self.button3.bind(on_press=self.go_back_button)
        self.add_widget(self.button3)
        
        self.button4 = Button(text="Ana Menü",background_normal="buttons/home_button.png", font_size=35)
        self.button4.bind(on_press=self.main_menu_button)
        self.add_widget(self.button4)
        
        Clock.schedule_interval(lambda dt: self.callback_f(), GLOBAL_TIMER_VALUE)
        
        self.update_texts()
    

    def call_button(self, instances):
        self.set_current_screen(False)
        main_app.call_page.update_texts(self.call_info, "")
        main_app.screen_manager.current = "Call"
        main_app.call_page.set_current_screen(True)

    def message_button(self, instances):
        self.set_current_screen(False)
        main_app.message_page.info2 = self.call_info
        main_app.screen_manager.current = "Message"
        main_app.message_page.set_current_screen(True)
        
    def go_back_button(self, instances):
        self.set_current_screen(False)
        main_app.screen_manager.current = "Contacts"
        main_app.contacts_page.set_current_screen(True)

    def main_menu_button(self, instances):
        self.set_current_screen(False)
        main_app.screen_manager.current = "Main"
        main_app.main_page.set_current_screen(True)
    
    
    def next_option(self):
        if(self.current_option < self.button_rows-1):
            self.current_option += 1
            self.update_texts()
            
    def previous_option(self):
        if(self.current_option > 0):
            self.current_option -= 1
            self.update_texts()
    
    def update_texts(self):
        if(self.current_option == 0):
            self.button1.background_color = COLOR_CHOOSEN
        else:
            self.button1.background_color = COLOR_OTHERS
            
        if(self.current_option == 1):
            self.button2.background_color = COLOR_CHOOSEN
        else:
            self.button2.background_color = COLOR_OTHERS
            
        if(self.current_option == 2):
            self.button3.background_color = COLOR_CHOOSEN
        else:
            self.button3.background_color = COLOR_OTHERS
        
        if(self.current_option == 3):
            self.button4.background_color = COLOR_CHOOSEN
        else:
            self.button4.background_color = COLOR_OTHERS
    
    def choose_current_option(self):
        if(self.current_option == 0):
            self.call_button(1)
        elif(self.current_option == 1):
            self.message_button(1)
        elif(self.current_option == 2):
            self.go_back_button(1)
        elif(self.current_option == 3):
            self.main_menu_button(1)
        else:
            print("ERROR - current_option = ", self.current_option)
    
    def set_current_screen(self, status):
        self.current_screen = status
    
    def set_call_info(self, info):
        self.call_info = info
    
    def callback_f(self):
        if(self.current_screen):
            read_data()
            if len(data_dict['rawValue']) > 100:
                blink = find_peak(np.array(data_dict['rawValue'])[:100])
                if blink == 1:
                    self.choose_current_option()
                testData = []
                for data in featureList:
                    testData.append(np.array(data_dict[data])[:100])
                testData = np.array(testData)
                predict(model,testData)
                
                init_DataDict()
                #    for _ in range(100):
                #       data_dict[data].popleft()
                if(self.current_option == self.button_rows-1):
                    self.current_option = -1
                self.next_option()


class CallPage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.cols = 1
        self.rows = 3
        self.button_rows = 11
        
        self.current_option = 0
        self.current_screen = False
        
        self.text1 = "\nAranıyor"
        self.text2 = ""
        
        
        self.label1 = Button(text=self.text1,background_normal="buttons/e_button.png", font_size=35)
        self.add_widget(self.label1)
        
        self.label2 = Button(text=self.text2,background_normal="buttons/e_button.png", font_size=35)
        self.add_widget(self.label2)
        
        
        
        self.button1 = Button(text="Ana Menü",background_normal="buttons/home_button.png")
        self.button1.bind(on_press=self.main_menu_button)
        self.add_widget(self.button1)      
        
        Clock.schedule_interval(lambda dt: self.callback_f(), GLOBAL_TIMER_VALUE)
        
        #self.update_texts(self.text1, self.text2)
        
        


    def main_menu_button(self, instances):
        self.set_current_screen(False)
        main_app.screen_manager.current = "Main"
        main_app.main_page.set_current_screen(True)
        
    """
    def next_option(self):
        if(self.current_option < self.button_rows-1):
            self.current_option += 1
            self.update_texts()
            
    def previous_option(self):
        if(self.current_option > 0):
            self.current_option -= 1
            self.update_texts()
    """
    def update_texts(self, new_text1, new_text2):
        self.label1.text = new_text1 + self.text1
        self.label2.text = self.text2 + new_text2
    
        if(self.current_option == 0):
            self.button1.background_color = COLOR_CHOOSEN
        else:
            self.button1.background_color = COLOR_OTHERS
            
        
            
    
    def choose_current_option(self):
        if(self.current_option == 0):
            self.main_menu_button(1)
        else:
            print("ERROR - current_option = ", self.current_option)
    
    def set_current_screen(self, status):
        self.current_screen = status
    
    def callback_f(self):
        if(self.current_screen):
            read_data()
            if len(data_dict['rawValue']) > 100:
                blink = find_peak(np.array(data_dict['rawValue'])[:100])
                if blink == 1:
                    self.main_menu_button(1)
                testData = []
                for data in featureList:
                    testData.append(np.array(data_dict[data])[:100])
                testData = np.array(testData)
                predict(model,testData)
                
                init_DataDict()
                #    for _ in range(100):
                #       data_dict[data].popleft()
                
                

    
class MessagePage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.cols = 1
        self.rows = 4
        self.button_rows = 4
        
        self.current_option = 0
        self.current_screen = False
        
        self.info1 = ""
        self.info2 = ""
        
        self.message_text_1 = "Mesaj 1 - Merhaba"
        self.message_text_2 = "Mesaj 2 - Yardım"
        self.message_text_3 = "Mesaj 3 - Saat 1'de öğle yemeği :)"
        
        self.button1 = Button(text=self.message_text_1,background_normal="buttons/e_button.png", font_size=35)
        self.button1.bind(on_press=self.send_button)
        self.add_widget(self.button1)
        
        self.button2 = Button(text=self.message_text_2,background_normal="buttons/e_button.png", font_size=35)
        self.button2.bind(on_press=self.send_button)
        self.add_widget(self.button2)
        
        self.button3 = Button(text=self.message_text_3,background_normal="buttons/e_button.png", font_size=35) 
        self.button3.bind(on_press=self.send_button)
        self.add_widget(self.button3)
        
        
        self.button4 = Button(text="Ana Menü",background_normal="buttons/home_button.png", font_size=35)
        self.button4.bind(on_press=self.main_menu_button)
        self.add_widget(self.button4)

        Clock.schedule_interval(lambda dt: self.callback_f(), GLOBAL_TIMER_VALUE)
        
        self.update_texts()
        

    def send_button(self, instances):
        if(self.current_option == 0):
            self.info1 = self.message_text_1[10:]
        elif(self.current_option == 1):
            self.info1 = self.message_text_2[10:]
        elif(self.current_option == 2):
            self.info1 = self.message_text_3[10:]
        self.set_current_screen(False)
        main_app.message_sent_page.update_texts(self.info1, self.info2)
        main_app.screen_manager.current = "Message_Sent"
        main_app.message_sent_page.set_current_screen(True)
    

    def main_menu_button(self, instances):
        self.set_current_screen(False)
        main_app.screen_manager.current = "Main"
        main_app.main_page.set_current_screen(True)
        
    
    def next_option(self):
        if(self.current_option < self.button_rows-1):
            self.current_option += 1
            self.update_texts()
            
    def previous_option(self):
        if(self.current_option > 0):
            self.current_option -= 1
            self.update_texts()
    
    def update_texts(self):
        if(self.current_option == 0):
            self.button1.background_color = COLOR_CHOOSEN
        else:
            self.button1.background_color = COLOR_OTHERS
            
        if(self.current_option == 1):
            self.button2.background_color = COLOR_CHOOSEN
        else:
            self.button2.background_color = COLOR_OTHERS
            
        if(self.current_option == 2):
            self.button3.background_color = COLOR_CHOOSEN
        else:
            self.button3.background_color = COLOR_OTHERS
        
        if(self.current_option == 3):
            self.button4.background_color = COLOR_CHOOSEN
        else:
            self.button4.background_color = COLOR_OTHERS
    
    def choose_current_option(self):
        if(self.current_option == 0):
            self.send_button(1)
        elif(self.current_option == 1):
            self.send_button(1)
        elif(self.current_option == 2):
            self.send_button(1)
        elif(self.current_option == 3):
            self.main_menu_button(1)
        else:
            print("ERROR - current_option = ", self.current_option)
    
    def set_current_screen(self, status):
        self.current_screen = status
    
    def callback_f(self):
        if(self.current_screen):
            read_data()
            if len(data_dict['rawValue']) > 100:
                blink = find_peak(np.array(data_dict['rawValue'])[:100])
                if blink == 1:
                    self.choose_current_option()
                testData = []
                for data in featureList:
                    testData.append(np.array(data_dict[data])[:100])
                testData = np.array(testData)
                predict(model,testData)
                
                init_DataDict()
                #    for _ in range(100):
                #       data_dict[data].popleft()
                if(self.current_option == self.button_rows-1):
                    self.current_option = -1
                self.next_option()


class MessageSentPage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.cols = 1
        self.rows = 3
        self.button_rows = 1
        
        self.current_option = 0
        self.current_screen = False
        
        self.text1 = "Mesaj: "
        self.text2 = "Kime: "
        
        
        self.label1 = Button(text=self.text1,background_normal="buttons/e_button.png", font_size=35)
        self.add_widget(self.label1)
        
        self.label2 = Button(text=self.text2,background_normal="buttons/e_button.png", font_size=35)
        self.add_widget(self.label2)
        
        
        self.button1 = Button(text="Ana Menü",background_normal="buttons/home_button.png", font_size=35)
        self.button1.bind(on_press=self.main_menu_button)
        self.add_widget(self.button1)
        
        
        Clock.schedule_interval(lambda dt: self.callback_f(), GLOBAL_TIMER_VALUE)
        
        #self.update_texts(self.text1, self.text2)
    
    

    def main_menu_button(self, instances):
        self.set_current_screen(False)
        main_app.screen_manager.current = "Main"
        main_app.main_page.set_current_screen(True)
    
    """
    def next_option(self):
        if(self.current_option < self.button_rows-1):
            self.current_option += 1
            self.update_texts()
            
    def previous_option(self):
        if(self.current_option > 0):
            self.current_option -= 1
            self.update_texts()
    """
    def update_texts(self, new_text1, new_text2):
        print(new_text1, new_text2)
        self.label1.text = self.text1 + new_text1
        self.label2.text = self.text2 + new_text2 + "\n\nGönderildi"
    
        if(self.current_option == 0):
            self.button1.background_color = COLOR_CHOOSEN
        else:
            self.button1.background_color = COLOR_OTHERS
            
        
            
    
    def choose_current_option(self):
        if(self.current_option == 0):
            self.main_menu_button(1)
        else:
            print("ERROR - current_option = ", self.current_option)
    
    def set_current_screen(self, status):
        self.current_screen = status
    
    def callback_f(self):
        if(self.current_screen):
            read_data()
            if len(data_dict['rawValue']) > 100:
                blink = find_peak(np.array(data_dict['rawValue'])[:100])
                if blink == 1:
                    self.main_menu_button(1)
                testData = []
                for data in featureList:
                    testData.append(np.array(data_dict[data])[:100])
                testData = np.array(testData)
                predict(model,testData)
                
                init_DataDict()
                #    for _ in range(100):
                #       data_dict[data].popleft()
                
                
    

class KeyboardPage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.cols = 1
        self.rows = 2
        self.button_rows = 1
        
        self.label_text = ""
        
        self.current_row = 0
        self.current_col = 4
        
        self.selecting_cols = True
        
        self.current_screen = False
        
        
        self.label1 = Label(text=self.label_text, size_hint_y=None, height=100)
        self.add_widget(self.label1)
        
        
        self.key_layout = GridLayout(rows=5, cols=5)
        self.add_widget(self.key_layout)
        
        self.button11 = Button(text="A")
        self.button11.bind(on_press=self.button11_f)
        self.key_layout.add_widget(self.button11)
        
        self.button12 = Button(text="B")
        self.button12.bind(on_press=self.button12_f)
        self.key_layout.add_widget(self.button12)
        
        self.button13 = Button(text="C")
        self.button13.bind(on_press=self.button13_f)
        self.key_layout.add_widget(self.button13)
        
        self.button14 = Button(text="D")
        self.button14.bind(on_press=self.button14_f)
        self.key_layout.add_widget(self.button14)

        self.button15 = Button(text="E")
        self.button15.bind(on_press=self.button15_f)
        self.key_layout.add_widget(self.button15)


        self.button21 = Button(text="F")
        self.button21.bind(on_press=self.button21_f)
        self.key_layout.add_widget(self.button21)
        
        self.button22 = Button(text="G")
        self.button22.bind(on_press=self.button22_f)
        self.key_layout.add_widget(self.button22)
        
        self.button23 = Button(text="H")
        self.button23.bind(on_press=self.button23_f)
        self.key_layout.add_widget(self.button23)
        
        self.button24 = Button(text="I")
        self.button24.bind(on_press=self.button24_f)
        self.key_layout.add_widget(self.button24)

        self.button25 = Button(text="J")
        self.button25.bind(on_press=self.button25_f)
        self.key_layout.add_widget(self.button25)


        self.button31 = Button(text="K")
        self.button31.bind(on_press=self.button31_f)
        self.key_layout.add_widget(self.button31)
        
        self.button32 = Button(text="L")
        self.button32.bind(on_press=self.button32_f)
        self.key_layout.add_widget(self.button32)

        self.button33 = Button(text="M")
        self.button33.bind(on_press=self.button33_f)
        self.key_layout.add_widget(self.button33)

        self.button34 = Button(text="N")
        self.button34.bind(on_press=self.button34_f)
        self.key_layout.add_widget(self.button34)

        self.button35 = Button(text="O")
        self.button35.bind(on_press=self.button35_f)
        self.key_layout.add_widget(self.button35)


        self.button41 = Button(text="P")
        self.button41.bind(on_press=self.button41_f)
        self.key_layout.add_widget(self.button41)
        
        self.button42 = Button(text="R")
        self.button42.bind(on_press=self.button42_f)
        self.key_layout.add_widget(self.button42)

        self.button43 = Button(text="S")
        self.button43.bind(on_press=self.button43_f)
        self.key_layout.add_widget(self.button43)

        self.button44 = Button(text="T")
        self.button44.bind(on_press=self.button44_f)
        self.key_layout.add_widget(self.button44)

        self.button45 = Button(text="U")
        self.button45.bind(on_press=self.button45_f)
        self.key_layout.add_widget(self.button45)


        self.button51 = Button(text="V")
        self.button51.bind(on_press=self.button51_f)
        self.key_layout.add_widget(self.button51)

        self.button52 = Button(text="Y")
        self.button52.bind(on_press=self.button52_f)
        self.key_layout.add_widget(self.button52)

        self.button53 = Button(text="Z")
        self.button53.bind(on_press=self.button53_f)
        self.key_layout.add_widget(self.button53)

        self.button54 = Button(text="[Space]")
        self.button54.bind(on_press=self.button54_f)
        self.key_layout.add_widget(self.button54)

        
        self.button55 = Button(text="Ana Menü")
        self.button55.bind(on_press=self.button55_f)
        self.key_layout.add_widget(self.button55)
        
        Clock.schedule_interval(lambda dt: self.callback_f(), GLOBAL_TIMER_VALUE)
        
    def change_selecting_option(self):
        if(self.selecting_cols):
            self.selecting_cols = False
            self.current_row = 4
        else:
            self.selecting_cols = True
            self.current_row = 4
            self.current_col = 4
        
    def button11_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "A"
            self.label1.text = self.label_text
        self.change_selecting_option()
    def button12_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "B"
            self.label1.text = self.label_text
        self.change_selecting_option()
    def button13_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "C"
            self.label1.text = self.label_text
        self.change_selecting_option()
    def button14_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "D"
            self.label1.text = self.label_text
        self.change_selecting_option()
    def button15_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "E"
            self.label1.text = self.label_text
        self.change_selecting_option()

    def button21_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "F"
            self.label1.text = self.label_text
        self.change_selecting_option()
    def button22_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "G"
            self.label1.text = self.label_text
        self.change_selecting_option()
    def button23_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "H"
            self.label1.text = self.label_text
        self.change_selecting_option()
    def button24_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "I"
            self.label1.text = self.label_text
        self.change_selecting_option()
    def button25_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "J"
            self.label1.text = self.label_text
        self.change_selecting_option()

    def button31_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "K"
            self.label1.text = self.label_text
        self.change_selecting_option()
    def button32_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "L"
            self.label1.text = self.label_text
        self.change_selecting_option()
    def button33_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "M"
            self.label1.text = self.label_text
        self.change_selecting_option()
    def button34_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "N"
            self.label1.text = self.label_text
        self.change_selecting_option()
    def button35_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "O"
            self.label1.text = self.label_text
        self.change_selecting_option()

    def button41_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "P"
            self.label1.text = self.label_text
        self.change_selecting_option()
    def button42_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "R"
            self.label1.text = self.label_text
        self.change_selecting_option()
    def button43_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "S"
            self.label1.text = self.label_text
        self.change_selecting_option()
    def button44_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "T"
            self.label1.text = self.label_text
        self.change_selecting_option()
    def button45_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "U"
            self.label1.text = self.label_text
        self.change_selecting_option()

    def button51_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "V"
            self.label1.text = self.label_text
        self.change_selecting_option()
    def button52_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "Y"
            self.label1.text = self.label_text
        self.change_selecting_option()
    def button53_f(self, instances):
        if not self.selecting_cols:
            self.label_text += "Z"
            self.label1.text = self.label_text
        self.change_selecting_option()
    def button54_f(self, instances):
        if not self.selecting_cols:
            self.label_text += " "
            self.label1.text = self.label_text
        self.change_selecting_option()


    def button55_f(self, instances):
        if not self.selecting_cols:
            self.selecting_cols = True
            self.set_current_screen(False)
            main_app.screen_manager.current = "Main"
            main_app.main_page.set_current_screen(True)
            self.label_text = ""
        self.change_selecting_option()


    def main_menu_button(self, instances):
        if not self.selecting_cols:
            self.selecting_cols = True
            self.set_current_screen(False)
            main_app.screen_manager.current = "Main"
            main_app.main_page.set_current_screen(True)
            self.label_text = ""
        self.change_selecting_option()



    def choose_current_option(self):
        if(self.current_row == 0):
            if(self.current_col == 0):
                self.button11_f(1)
            elif(self.current_col == 1):
                self.button12_f(1)
            elif(self.current_col == 2):
                self.button13_f(1)
            elif(self.current_col == 3):
                self.button14_f(1)
            elif(self.current_col == 4):
                self.button15_f(1)
            else:
                print("HHHHHAAAAATTTTTAAAA", self.current_row, self.current_col)
        elif(self.current_row == 1):
            if(self.current_col == 0):
                self.button21_f(1)
            elif(self.current_col == 1):
                self.button22_f(1)
            elif(self.current_col == 2):
                self.button23_f(1)
            elif(self.current_col == 3):
                self.button24_f(1)
            elif(self.current_col == 4):
                self.button25_f(1)
            else:
                print("HHHHHAAAAATTTTTAAAA", self.current_row, self.current_col)
        elif(self.current_row == 2):
            if(self.current_col == 0):
                self.button31_f(1)
            elif(self.current_col == 1):
                self.button32_f(1)
            elif(self.current_col == 2):
                self.button33_f(1)
            elif(self.current_col == 3):
                self.button34_f(1)
            elif(self.current_col == 4):
                self.button35_f(1)
            else:
                print("HHHHHAAAAATTTTTAAAA", self.current_row, self.current_col)
        elif(self.current_row == 3):
            if(self.current_col == 0):
                self.button41_f(1)
            elif(self.current_col == 1):
                self.button42_f(1)
            elif(self.current_col == 2):
                self.button43_f(1)
            elif(self.current_col == 3):
                self.button44_f(1)
            elif(self.current_col == 4):
                self.button45_f(1)
            else:
                print("HHHHHAAAAATTTTTAAAA", self.current_row, self.current_col)
        elif(self.current_row == 4):
            if(self.current_col == 0):
                self.button51_f(1)
            elif(self.current_col == 1):
                self.button52_f(1)
            elif(self.current_col == 2):
                self.button53_f(1)
            elif(self.current_col == 3):
                self.button54_f(1)
            elif(self.current_col == 4):
                self.button55_f(1)
            else:
                print("HHHHHAAAAATTTTTAAAA", self.current_row, self.current_col)
        else:
            print("HHHHHAAAAATTTTTAAAA", self.current_row, self.current_col)
        


    
    def update_for_cols(self):
        if(self.current_col == 0):
            self.button11.background_color = COLOR_CHOOSEN
            self.button21.background_color = COLOR_CHOOSEN
            self.button31.background_color = COLOR_CHOOSEN
            self.button41.background_color = COLOR_CHOOSEN
            self.button51.background_color = COLOR_CHOOSEN
            self.button12.background_color = COLOR_OTHERS
            self.button22.background_color = COLOR_OTHERS
            self.button32.background_color = COLOR_OTHERS
            self.button42.background_color = COLOR_OTHERS
            self.button52.background_color = COLOR_OTHERS
            self.button13.background_color = COLOR_OTHERS
            self.button23.background_color = COLOR_OTHERS
            self.button33.background_color = COLOR_OTHERS
            self.button43.background_color = COLOR_OTHERS
            self.button53.background_color = COLOR_OTHERS
            self.button14.background_color = COLOR_OTHERS
            self.button24.background_color = COLOR_OTHERS
            self.button34.background_color = COLOR_OTHERS
            self.button44.background_color = COLOR_OTHERS
            self.button54.background_color = COLOR_OTHERS
            self.button15.background_color = COLOR_OTHERS
            self.button25.background_color = COLOR_OTHERS
            self.button35.background_color = COLOR_OTHERS
            self.button45.background_color = COLOR_OTHERS
            self.button55.background_color = COLOR_OTHERS
        elif(self.current_col == 1):
            self.button12.background_color = COLOR_CHOOSEN
            self.button22.background_color = COLOR_CHOOSEN
            self.button32.background_color = COLOR_CHOOSEN
            self.button42.background_color = COLOR_CHOOSEN
            self.button52.background_color = COLOR_CHOOSEN
            self.button11.background_color = COLOR_OTHERS
            self.button21.background_color = COLOR_OTHERS
            self.button31.background_color = COLOR_OTHERS
            self.button41.background_color = COLOR_OTHERS
            self.button51.background_color = COLOR_OTHERS
            self.button13.background_color = COLOR_OTHERS
            self.button23.background_color = COLOR_OTHERS
            self.button33.background_color = COLOR_OTHERS
            self.button43.background_color = COLOR_OTHERS
            self.button53.background_color = COLOR_OTHERS
            self.button14.background_color = COLOR_OTHERS
            self.button24.background_color = COLOR_OTHERS
            self.button34.background_color = COLOR_OTHERS
            self.button44.background_color = COLOR_OTHERS
            self.button54.background_color = COLOR_OTHERS
            self.button15.background_color = COLOR_OTHERS
            self.button25.background_color = COLOR_OTHERS
            self.button35.background_color = COLOR_OTHERS
            self.button45.background_color = COLOR_OTHERS
            self.button55.background_color = COLOR_OTHERS
        elif(self.current_col == 2):
            self.button13.background_color = COLOR_CHOOSEN
            self.button23.background_color = COLOR_CHOOSEN
            self.button33.background_color = COLOR_CHOOSEN
            self.button43.background_color = COLOR_CHOOSEN
            self.button53.background_color = COLOR_CHOOSEN
            self.button11.background_color = COLOR_OTHERS
            self.button21.background_color = COLOR_OTHERS
            self.button31.background_color = COLOR_OTHERS
            self.button41.background_color = COLOR_OTHERS
            self.button51.background_color = COLOR_OTHERS
            self.button12.background_color = COLOR_OTHERS
            self.button22.background_color = COLOR_OTHERS
            self.button32.background_color = COLOR_OTHERS
            self.button42.background_color = COLOR_OTHERS
            self.button52.background_color = COLOR_OTHERS
            self.button14.background_color = COLOR_OTHERS
            self.button24.background_color = COLOR_OTHERS
            self.button34.background_color = COLOR_OTHERS
            self.button44.background_color = COLOR_OTHERS
            self.button54.background_color = COLOR_OTHERS
            self.button15.background_color = COLOR_OTHERS
            self.button25.background_color = COLOR_OTHERS
            self.button35.background_color = COLOR_OTHERS
            self.button45.background_color = COLOR_OTHERS
            self.button55.background_color = COLOR_OTHERS
        elif(self.current_col == 3):
            self.button14.background_color = COLOR_CHOOSEN
            self.button24.background_color = COLOR_CHOOSEN
            self.button34.background_color = COLOR_CHOOSEN
            self.button44.background_color = COLOR_CHOOSEN
            self.button54.background_color = COLOR_CHOOSEN
            self.button11.background_color = COLOR_OTHERS
            self.button21.background_color = COLOR_OTHERS
            self.button31.background_color = COLOR_OTHERS
            self.button41.background_color = COLOR_OTHERS
            self.button51.background_color = COLOR_OTHERS
            self.button12.background_color = COLOR_OTHERS
            self.button22.background_color = COLOR_OTHERS
            self.button32.background_color = COLOR_OTHERS
            self.button42.background_color = COLOR_OTHERS
            self.button52.background_color = COLOR_OTHERS
            self.button13.background_color = COLOR_OTHERS
            self.button23.background_color = COLOR_OTHERS
            self.button33.background_color = COLOR_OTHERS
            self.button43.background_color = COLOR_OTHERS
            self.button53.background_color = COLOR_OTHERS
            self.button15.background_color = COLOR_OTHERS
            self.button25.background_color = COLOR_OTHERS
            self.button35.background_color = COLOR_OTHERS
            self.button45.background_color = COLOR_OTHERS
            self.button55.background_color = COLOR_OTHERS
        elif(self.current_col == 4):
            self.button15.background_color = COLOR_CHOOSEN
            self.button25.background_color = COLOR_CHOOSEN
            self.button35.background_color = COLOR_CHOOSEN
            self.button45.background_color = COLOR_CHOOSEN
            self.button55.background_color = COLOR_CHOOSEN
            self.button11.background_color = COLOR_OTHERS
            self.button21.background_color = COLOR_OTHERS
            self.button31.background_color = COLOR_OTHERS
            self.button41.background_color = COLOR_OTHERS
            self.button51.background_color = COLOR_OTHERS
            self.button12.background_color = COLOR_OTHERS
            self.button22.background_color = COLOR_OTHERS
            self.button32.background_color = COLOR_OTHERS
            self.button42.background_color = COLOR_OTHERS
            self.button52.background_color = COLOR_OTHERS
            self.button13.background_color = COLOR_OTHERS
            self.button23.background_color = COLOR_OTHERS
            self.button33.background_color = COLOR_OTHERS
            self.button43.background_color = COLOR_OTHERS
            self.button53.background_color = COLOR_OTHERS
            self.button14.background_color = COLOR_OTHERS
            self.button24.background_color = COLOR_OTHERS
            self.button34.background_color = COLOR_OTHERS
            self.button44.background_color = COLOR_OTHERS
            self.button54.background_color = COLOR_OTHERS
    
    def update_for_rows(self):
        if(self.current_col == 0):
            if(self.current_row == 0):
                self.button11.background_color = COLOR_CHOOSEN
                self.button21.background_color = COLOR_OTHERS
                self.button31.background_color = COLOR_OTHERS
                self.button41.background_color = COLOR_OTHERS
                self.button51.background_color = COLOR_OTHERS
            elif(self.current_row == 1):
                self.button21.background_color = COLOR_CHOOSEN
                self.button11.background_color = COLOR_OTHERS
                self.button31.background_color = COLOR_OTHERS
                self.button41.background_color = COLOR_OTHERS
                self.button51.background_color = COLOR_OTHERS
            elif(self.current_row == 2):
                self.button31.background_color = COLOR_CHOOSEN
                self.button11.background_color = COLOR_OTHERS
                self.button21.background_color = COLOR_OTHERS
                self.button41.background_color = COLOR_OTHERS
                self.button51.background_color = COLOR_OTHERS
            elif(self.current_row == 3):
                self.button41.background_color = COLOR_CHOOSEN
                self.button11.background_color = COLOR_OTHERS
                self.button21.background_color = COLOR_OTHERS
                self.button31.background_color = COLOR_OTHERS
                self.button51.background_color = COLOR_OTHERS
            elif(self.current_row == 4):
                self.button51.background_color = COLOR_CHOOSEN
                self.button11.background_color = COLOR_OTHERS
                self.button21.background_color = COLOR_OTHERS
                self.button31.background_color = COLOR_OTHERS
                self.button41.background_color = COLOR_OTHERS
        elif(self.current_col == 1):
            if(self.current_row == 0):
                self.button12.background_color = COLOR_CHOOSEN
                self.button22.background_color = COLOR_OTHERS
                self.button32.background_color = COLOR_OTHERS
                self.button42.background_color = COLOR_OTHERS
                self.button52.background_color = COLOR_OTHERS
            elif(self.current_row == 1):
                self.button22.background_color = COLOR_CHOOSEN
                self.button12.background_color = COLOR_OTHERS
                self.button32.background_color = COLOR_OTHERS
                self.button42.background_color = COLOR_OTHERS
                self.button52.background_color = COLOR_OTHERS
            elif(self.current_row == 2):
                self.button32.background_color = COLOR_CHOOSEN
                self.button12.background_color = COLOR_OTHERS
                self.button22.background_color = COLOR_OTHERS
                self.button42.background_color = COLOR_OTHERS
                self.button52.background_color = COLOR_OTHERS
            elif(self.current_row == 3):
                self.button42.background_color = COLOR_CHOOSEN
                self.button12.background_color = COLOR_OTHERS
                self.button22.background_color = COLOR_OTHERS
                self.button32.background_color = COLOR_OTHERS
                self.button52.background_color = COLOR_OTHERS
            elif(self.current_row == 4):
                self.button52.background_color = COLOR_CHOOSEN
                self.button12.background_color = COLOR_OTHERS
                self.button22.background_color = COLOR_OTHERS
                self.button32.background_color = COLOR_OTHERS
                self.button42.background_color = COLOR_OTHERS
        elif(self.current_col == 2):
            if(self.current_row == 0):
                self.button13.background_color = COLOR_CHOOSEN
                self.button23.background_color = COLOR_OTHERS
                self.button33.background_color = COLOR_OTHERS
                self.button43.background_color = COLOR_OTHERS
                self.button53.background_color = COLOR_OTHERS
            elif(self.current_row == 1):
                self.button23.background_color = COLOR_CHOOSEN
                self.button13.background_color = COLOR_OTHERS
                self.button33.background_color = COLOR_OTHERS
                self.button43.background_color = COLOR_OTHERS
                self.button53.background_color = COLOR_OTHERS
            elif(self.current_row == 2):
                self.button33.background_color = COLOR_CHOOSEN
                self.button13.background_color = COLOR_OTHERS
                self.button23.background_color = COLOR_OTHERS
                self.button43.background_color = COLOR_OTHERS
                self.button53.background_color = COLOR_OTHERS
            elif(self.current_row == 3):
                self.button43.background_color = COLOR_CHOOSEN
                self.button13.background_color = COLOR_OTHERS
                self.button23.background_color = COLOR_OTHERS
                self.button33.background_color = COLOR_OTHERS
                self.button53.background_color = COLOR_OTHERS
            elif(self.current_row == 4):
                self.button53.background_color = COLOR_CHOOSEN
                self.button13.background_color = COLOR_OTHERS
                self.button23.background_color = COLOR_OTHERS
                self.button33.background_color = COLOR_OTHERS
                self.button43.background_color = COLOR_OTHERS
        elif(self.current_col == 3):
            if(self.current_row == 0):
                self.button14.background_color = COLOR_CHOOSEN
                self.button24.background_color = COLOR_OTHERS
                self.button34.background_color = COLOR_OTHERS
                self.button44.background_color = COLOR_OTHERS
                self.button54.background_color = COLOR_OTHERS
            elif(self.current_row == 1):
                self.button24.background_color = COLOR_CHOOSEN
                self.button14.background_color = COLOR_OTHERS
                self.button34.background_color = COLOR_OTHERS
                self.button44.background_color = COLOR_OTHERS
                self.button54.background_color = COLOR_OTHERS
            elif(self.current_row == 2):
                self.button34.background_color = COLOR_CHOOSEN
                self.button14.background_color = COLOR_OTHERS
                self.button24.background_color = COLOR_OTHERS
                self.button44.background_color = COLOR_OTHERS
                self.button54.background_color = COLOR_OTHERS
            elif(self.current_row == 3):
                self.button44.background_color = COLOR_CHOOSEN
                self.button14.background_color = COLOR_OTHERS
                self.button24.background_color = COLOR_OTHERS
                self.button34.background_color = COLOR_OTHERS
                self.button54.background_color = COLOR_OTHERS
            elif(self.current_row == 4):
                self.button54.background_color = COLOR_CHOOSEN
                self.button14.background_color = COLOR_OTHERS
                self.button24.background_color = COLOR_OTHERS
                self.button34.background_color = COLOR_OTHERS
                self.button44.background_color = COLOR_OTHERS
        elif(self.current_col == 4):
            if(self.current_row == 0):
                self.button15.background_color = COLOR_CHOOSEN
                self.button25.background_color = COLOR_OTHERS
                self.button35.background_color = COLOR_OTHERS
                self.button45.background_color = COLOR_OTHERS
                self.button55.background_color = COLOR_OTHERS
            elif(self.current_row == 1):
                self.button25.background_color = COLOR_CHOOSEN
                self.button15.background_color = COLOR_OTHERS
                self.button35.background_color = COLOR_OTHERS
                self.button45.background_color = COLOR_OTHERS
                self.button55.background_color = COLOR_OTHERS
            elif(self.current_row == 2):
                self.button35.background_color = COLOR_CHOOSEN
                self.button15.background_color = COLOR_OTHERS
                self.button25.background_color = COLOR_OTHERS
                self.button45.background_color = COLOR_OTHERS
                self.button55.background_color = COLOR_OTHERS
            elif(self.current_row == 3):
                self.button45.background_color = COLOR_CHOOSEN
                self.button15.background_color = COLOR_OTHERS
                self.button25.background_color = COLOR_OTHERS
                self.button35.background_color = COLOR_OTHERS
                self.button55.background_color = COLOR_OTHERS
            elif(self.current_row == 4):
                self.button55.background_color = COLOR_CHOOSEN
                self.button15.background_color = COLOR_OTHERS
                self.button25.background_color = COLOR_OTHERS
                self.button35.background_color = COLOR_OTHERS
                self.button45.background_color = COLOR_OTHERS
            
    
    
    def set_current_screen(self, status):
        self.current_screen = status
        if(status):
            self.label_text = ""
            self.label1.text = self.label_text
        
            self.current_row = 0
            self.current_col = 4
        
            self.selecting_cols = True

    
    def callback_f(self):
        if(self.current_screen):
            read_data()
            if len(data_dict['rawValue']) > 100:
                blink = find_peak(np.array(data_dict['rawValue'])[:100])
                if blink == 1:
                    self.choose_current_option()
                testData = []
                for data in featureList:
                    testData.append(np.array(data_dict[data])[:100])
                testData = np.array(testData)
                predict(model,testData)
                
                init_DataDict()
                #    for _ in range(100):
                #       data_dict[data].popleft()
                

                if(self.selecting_cols):
                    self.current_col += 1
                    if(self.current_col > 4):
                        self.current_col = 0
                    self.update_for_cols()
                else:
                    self.current_row += 1
                    if(self.current_row > 4):
                        self.current_row = 0
                    self.update_for_rows()
    
        


class MainApp(App):
    def build(self):
        self.screen_manager = ScreenManager()
        
        self.main_page = MainPage()
        screen = Screen(name="Main")
        screen.add_widget(self.main_page)
        self.screen_manager.add_widget(screen)

        self.keyboard_page = KeyboardPage()
        screen = Screen(name="Keyboard")
        screen.add_widget(self.keyboard_page)
        self.screen_manager.add_widget(screen)
        
        self.pong_page = PongPage()
        screen = Screen(name="Pong")
        screen.add_widget(self.pong_page)
        self.screen_manager.add_widget(screen)
        
        self.books_page = BooksPage()
        screen = Screen(name="Books")
        screen.add_widget(self.books_page)
        self.screen_manager.add_widget(screen)
        
        self.book_1_page = Book1Page()
        screen = Screen(name="Book1")
        screen.add_widget(self.book_1_page)
        self.screen_manager.add_widget(screen)
        
        self.book_2_page = Book2Page()
        screen = Screen(name="Book2")
        screen.add_widget(self.book_2_page)
        self.screen_manager.add_widget(screen)
        
        self.contacts_page = ContactsPage()
        screen = Screen(name="Contacts")
        screen.add_widget(self.contacts_page)
        self.screen_manager.add_widget(screen)
        
        self.action_page = ActionPage()
        screen = Screen(name="Action")
        screen.add_widget(self.action_page)
        self.screen_manager.add_widget(screen)
        
        self.call_page = CallPage()
        screen = Screen(name="Call")
        screen.add_widget(self.call_page)
        self.screen_manager.add_widget(screen)
        
        self.message_page = MessagePage()
        screen = Screen(name="Message")
        screen.add_widget(self.message_page)
        self.screen_manager.add_widget(screen)
        
        self.message_sent_page = MessageSentPage()
        screen = Screen(name="Message_Sent")
        screen.add_widget(self.message_sent_page)
        self.screen_manager.add_widget(screen)
        
        self.main_page.set_current_screen(True)
         
        
        return self.screen_manager
        


        

if __name__ == "__main__":
    main_app = MainApp()
    main_app.run()
    neuropy.stop()
    

    

    