int serialUpdateRate = 100; //シーリアル速度（ms）
int deltaTime = 13; //ms
float radius = 1; //半径

int vx = 0;
int vy = 0;
int r = 0;

struct Motor {
  public:
  int val = 0; //-100 to 100
  int change = 0; //d>=100 to change
  int phase = 0; //0 to 3
  int scale = 1;
  
  int p1, p2; //00 01 11 10
  void Init () {
    pinMode(p1, OUTPUT);
    pinMode(p2, OUTPUT);
    Set();
  }
  
  void Update () {
    change += val;
    if (change >= 100*scale) {
      phase++;
      change -= 100*scale;
      if (phase >= 4)
        phase -= 4;
      Set();
    }
    if (change <= -100*scale) {
      phase--;
      change += 100*scale;
      if (phase < 0)
        phase += 4;
      Set();
    }
  }

  void Set () {
    digitalWrite(p1, phase>1?HIGH:LOW);
    digitalWrite(p2, (phase==1 || phase==2)?HIGH:LOW);
  }
};

Motor m1, m2, m3;
float targetCounter = -1;
bool targetMove;

void EvalMotors () {
  /*test
  m1.val = vx;
  m2.val = vy;
  m3.val = r;
  */
  
  float r3o2 = sqrt(3)*0.5;
  
  ///*
  m1.val = radius*r - 0.5*vx - r3o2*vy;
  m2.val = radius*r - 0.5*vx + r3o2*vy;
  m3.val = radius*r + vx;
  //*/
}

void Bot_MoveA(bool isy, int sp) {
  vx = 0;
  vy = 0;
  Bot_Move(isy, sp);
}

void Bot_Move(bool isy, int sp) {
  if (isy)
    vy = -sp;
  else
    vx = -sp;
  EvalMotors();
}

void Bot_Rotate(int w) {
  r = w;
  EvalMotors();
}

void Bot_MoveV(int x, int y) {
  if (x > y) {
    vx = -100;
    vy = -(int)((y*100.0f)/(x*1.0f));
  }
  else {
    vy = -100;
    vx = -(int)((x*100.0f)/(y*1.0f));
  }
  r = 0;
  targetCounter = max(x, y);
  targetMove = true;
  EvalMotors();
}

void Bot_RotateV(int vw) {
  r = 100;
  vx = 0;
  vy = 0;
  targetCounter = vw*radius;
  targetMove = false;
  EvalMotors();
}


/* Code to communicate with Android Souzou1
 * Created by @chokomancarr
 * 
 * 絶対に（絶対に）Serialとdelay使わないでください
 * 
 * 実装する関数（実装しないと動かない
 * Init() setupに呼ぶ
 * Update() loopに呼ぶ
 * ＊全ての値は-100~100
 * Bot_Move(bool isy, int sp) 移動する（isy=左右？、sp=前後/左右移動速度 100=>30cm毎秒
 * Bot_Rotate(int w) 回転する（w=回転速度 100=>180度毎秒
 * 
 * 距離制コマンド（うまくいかない気がする( ˘ω˘)
 * Bot_MoveV(bool isy, int v) 移動する（isy=左右？、v=前後/左右移動距離 100=>100cm
 * Bot_RotateV(int vw) 回転する（vw=回転角度 100=>180度
 * 
 * Bot_◯◯Vの行動が終わったら以下の関数を読んでください
 * DoneRotate() 回転が終わった
 * DoneMove() 移動が終わった
 * 
 * 他に使える関数
 * Out(string s) デバッグに吐く（メッセ－ジが表示される）
 * 
 * 
 * 
 * 以下メモ（消さないで）
 * 
 * init 1111 1111 x3
 * data format 10ty_pCiI 0val_valv where I=id
 * rotate 0[000] 0000
 * move 0[001] 0000
 * invert value 0000 [1]000
 * isVal 0000 000[X] ack id=type id+1
 * 
 * complete action 1000_0[III] where I=id
 * log {ST chars ED} ascii only
 *  ST 1100 0000
 *  ED 0000 0011 (ETX)
 */
int waitingStSignal = 3;
int waitingType = -1;
bool waitingID = false;
bool waitingInv = false;
bool waitingCoupleX, waitingCoupleY;
int waitingCoupleVal;
unsigned long currentMillis = 0;
unsigned long lastMillis = 0;
void Init () {
  Serial.begin(9600);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
}

void Update () {
  currentMillis = millis();
  if (currentMillis - lastMillis > serialUpdateRate) {
    lastMillis = currentMillis;
    UpdateSerial();
  }
  //if (currentMillis - lastMillis > serialUpdateRate*0.5) 
    //digitalWrite(LED_BUILTIN, HIGH);
  //else
    //digitalWrite(LED_BUILTIN, LOW);
  //analogWrite(LED_BUILTIN, currentMillis-lastMillis);
}

void UpdateSerial () {
  int byteRead = 0;
  while (Serial.available() > 0) {
    byteRead = Serial.read();
    if (waitingStSignal <= 0) {
      if (byteRead == 255) //may remove
        continue;
      else if (byteRead >= 128) {
        waitingType = (byteRead >> 3) - 16;
        waitingInv = byteRead & 2 == 2;
        waitingID = byteRead & 1 == 1;

        if (!waitingCoupleX && !waitingCoupleY) {
          if (byteRead & 4 == 4) {
            waitingCoupleX = true;
            waitingCoupleY = true;
          }
        }
        else {
          if (byteRead & 4 == 4) {
            if (waitingType == 1)
              waitingCoupleY = true;
            if (waitingType == 2)
              waitingCoupleX = true;
          }
        }
        //digitalWrite(LED_BUILTIN, HIGH);
      }
      else {
        if (waitingType >= 0) {
          if (waitingType == 0) { //rotate
            if (waitingID)
              Bot_RotateV(waitingInv? -byteRead : byteRead);
            else
              Bot_Rotate(waitingInv? -byteRead : byteRead);
          }
          else if (waitingType == 1) { //translate y
            if (waitingID) {
              if (waitingCoupleY) {
                waitingCoupleY = false;
                if (!waitingCoupleX)
                  Bot_MoveV(waitingCoupleVal, waitingInv? -byteRead : byteRead);
                else
                  waitingCoupleVal = waitingInv? -byteRead : byteRead;
              }
              else 
                Bot_MoveV(false, waitingInv? -byteRead : byteRead);
            }
            else
              Bot_Move(false, waitingInv? -byteRead : byteRead);
          }
          else if (waitingType == 2) { //translate x
            if (waitingID) {
              if (waitingCoupleX) {
                waitingCoupleX = false;
                if (!waitingCoupleY)
                  Bot_MoveV(waitingInv? -byteRead : byteRead, waitingCoupleVal);
                else
                  waitingCoupleVal = waitingInv? -byteRead : byteRead;
              }
              else 
                Bot_MoveV(true, waitingInv? -byteRead : byteRead);
            }
            else
              Bot_Move(true, waitingInv? -byteRead : byteRead);
          }
        }
      }
    }
    else if (byteRead == 255) {
      waitingStSignal --;
      if (waitingStSignal <= 0)
        Out("ready");
    }
  }
}

void DoneRotate () {
  Serial.write((byte)(129));
}

void DoneMove () {
  Serial.write((byte)(131)); //130 131

void Out (String s) {
  if (waitingStSignal <= 0) {
    //char c[s.length()];
    //s.toCharArray(c, s.length());
    Serial.write((byte)(192));
    Serial.print(s);
    //Serial.print("z");
    Serial.write((byte)(3));
  }
}



//----------------------以下本文------------------------

void setup() {
  Init();
  m3.p1 = 2;
  m3.p2 = 3;
  m3.Init();
  m2.p1 = 4;
  m2.p2 = 5;
  m2.Init();
  m1.p1 = 6;
  m1.p2 = 7;
  m1.Init();
  //m1.val = 50;
  //m2.val = 50;
  //m3.val = -100;

  //Bot_Move(true, 50);
  //Bot_Rotate(50);
  
  pinMode(10, OUTPUT);

  //Bot_RotateV(200);
}

//delayの代わりにmillis使ってください
unsigned long timeNow = 0;
unsigned long timeLast = 0;
int a = 0;

unsigned long timeNow2 = 0;
unsigned long timeLast2 = 0;
int mm2 = 50;
void loop() {
  Update(); //一番上にのせて
  timeNow = millis();
  if (timeNow-timeLast > deltaTime) {
    timeLast = timeNow;
    //Out((a++)+"");
    
    m1.Update();
    m2.Update();
    m3.Update();

    if (targetCounter >= 0) {
      targetCounter -= (deltaTime*0.1f);
      if (targetCounter < 0) {
        if (targetMove) {
          DoneMove();
          Bot_MoveA(true, 0);
        }
        else {
          DoneRotate();
          Bot_Rotate(0);
        }
        EvalMotors();
      }
    }
  }

/*
  timeNow2 = millis();
  if (timeNow2-timeLast2 > 1000) {
    if (timeNow2-timeLast2 > 2000) {
      if (timeNow2-timeLast2 > 3000) {
        if (timeNow2-timeLast2 > 4000)
          timeLast2 = timeNow2;
        else
           Bot_MoveA(true, -70);
      }
      else
        Bot_MoveA(false, -70);
    }
    else {
      Bot_MoveA(true, 70);
    }
  }
  else
    Bot_MoveA(false, 70);
    /*/
  //analogWrite(11, (timeNow-timeLast)*0.2);
}
