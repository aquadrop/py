# memory
chatbot powered by memory network

## 思路
结合上下文将语句分类成几十万个类. 如果没有标记意图, 则一个回答就是一个类, 如果标记意图(组), 这组意图的所有回答是一个类

意图组可以转化成api_call

## 业务部分
业务部分仍然使用意图树来控制

## 需要考虑的问题
1. 识别价格
2. 识别空调匹数等一系列专业数值问题
3. 抽取实体(品类,地点,城市,公共设施等)

## request
* slot: machine_slot, user_slot, product_slot_list
* machine_slot: category, price, brand, TREE
* user_slot: 地点,名字,性别
* product_slot_list: deny_list, keep_list

# TASK
Task 1: Issuing API calls 

Task 2: Updating API calls Starting by issuing an API call as in Task 1

Task 3: Displaying options Given a user request

Task 4: Providing extra information Given a user request, we sample a restaurant and start the
dialog as if users had agreed to book a table there. 

Task 5: Conducting full dialogs We combine Tasks 1-4 to generate full dialogs just as in Figure 1.
Unlike in Task 3, we keep examples if API calls return at least 1 option instead of 3.

## example
### example -- course
```
user:你好 cls=api_call_greeting_search
machine: 你好,可以帮你什么? request:product_category
user: 我买空调 cls=购物,空调
machine: 什么牌子? request:brand  product_category=空调
user:格力 cls=格力
machine: 您的房间有多大? request: power product_category=空调,brand=格力
user: 40 cls=1.5(rule_based)
machine: 价位? request:price product_category=空调,brand=格力, power=1.5
user: 3000左右 cls=moderate(rule_based)
machine:为您推荐以下产品...xxxx product_category=空调, brand=格力, power=1.5,price=moderate
```
### example -- list_product_category
```
user:你好
machine: 你好,可以帮你什么? request:product_category
user: 你们这里有什么卖的? cls=api_call_list_product_category
machine: 我们这里有<xxx>等多种产品
user: 我买空调
machine: 什么牌子? request:brand  product_category=空调
user:格力
machine: 您的房间有多大? request: power product_category=空调,brand=格力
user: 40
machine: 价位? request:price product_category=空调,brand=格力, power=1.5
user: 3000左右
machine:为您推荐以下产品...xxxx product_category=空调, brand=格力, power=1.5,price=moderate
```

### example -- list_price
```
user:你好
machine: 你好,可以帮你什么? request:product_category
user: 你们这里有什么卖的? cls=api_call_list_product_category
machine: 我们这里有<xxx>等多种产品
user: 我买空调
machine: 什么牌子? request:brand  product_category=空调
user:格力
machine: 您的房间有多大? request: power product_category=空调,brand=格力
user: 40
machine: 价位? request:price product_category=空调,brand=格力, power=1.5
user: 你们有什么价位的///有哪些 cls=list_price
machine: 我们有1000-2000,2000-3000,....
machine:为您推荐以下产品...xxxx product_category=空调, brand=格力, power=1.5,price=moderate
```

### example3 insert_open_domain
```
user:你好 cls=api_call_greeting_search
machine: 你好,可以帮你什么? request:product_category
user: 你叫什么名字? cls=api_call_greeting_search
machine:我叫xxx
user:我买空调
machine: 我们这里有<xxx>等多种产品
user: 我买空调
machine: 什么牌子? request:brand  product_category=空调
user:格力
machine: 您的房间有多大? request: power product_category=空调,brand=格力
user: 40
machine: 价位? request:price product_category=空调,brand=格力, power=1.5
user: 3000左右
machine:为您推荐以下产品...xxxx product_category=空调, brand=格力, power=1.5,price=moderate
```
### example4 update_price
```
user:你好 cls=api_call_greeting_search
machine: 你好,可以帮你什么? request:product_category
user: 你叫什么名字? cls=api_call_greeting_search
machine:我叫xxx
user: 我买空调
machine: 什么牌子? request:brand  product_category=空调
user:格力
machine: 您的房间有多大? request: power product_category=空调,brand=格力
user: 40
machine: 价位? request:price product_category=空调,brand=格力, power=1.5
user: 3000左右
machine:为您推荐以下产品...xxxx product_category=空调, brand=格力, power=1.5,price=moderate
user: 太贵了 cls=update_price
machine: 你需要什么价位呢? request:price
user: 2000左右 cls=price:2000
machine: 为您推荐以下产品...xxx
```

### example5 list_more
```
user:你好 cls=api_call_greeting_search
machine: 你好,可以帮你什么? request:product_category
user: 你叫什么名字? cls=api_call_greeting_search
machine:我叫xxx
user: 我买空调
machine: 什么牌子? request:brand  product_category=空调
user:格力
machine: 您的房间有多大? request: power product_category=空调,brand=格力
user: 40
machine: 价位? request:price product_category=空调,brand=格力, power=1.5
user: 3000左右
machine:为您推荐以下产品...xxxx product_category=空调, brand=格力, power=1.5,price=moderate
user: 我不要这个 cls=list_more
machine: 还有如下产品xxx
```

### example6 ask_info
```
user:你好 cls=api_call_greeting_search
machine: 你好,可以帮你什么? request:product_category
user: 你叫什么名字? cls=api_call_greeting_search
machine:我叫xxx
user: 我买空调
machine: 什么牌子? request:brand  product_category=空调
user:格力
machine: 您的房间有多大? request: power product_category=空调,brand=格力
user: 40
machine: 价位? request:price product_category=空调,brand=格力, power=1.5
user: 3000左右
machine:为您推荐以下产品...xxxx product_category=空调, brand=格力, power=1.5,price=moderate
user: 在什么地方买 cls=ask_location
machine: 在三楼
user: 贵吗 cls=ask_price
machine: 不贵
user: 有打折吗? cls=ask_discount
machine: 目前没有打折
```

### example7 Ambiguity
```
user:你好 cls=api_call_greeting_search
machine: 你好,可以帮你什么? request:product_category
user: 你叫什么名字? cls=api_call_greeting_search
machine:我叫xxx
user: 苹果在什么地方买 cls=ask_location_苹果
machine:您是买手机还是水果
user: 手机? cls=手机 accumulate=ask_location_苹果,手机
machine: 在三楼买
user:我买梨子 cls=购物,水果
machine: 负一楼超市
```

### example8 other types
```
格力和美的哪个好
2000-3000的空调有哪些
```


