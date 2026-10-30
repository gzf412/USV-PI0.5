"""Publish heartbeat-protected motor intents to the episode recorder only."""
import signal
import json
import threading
import tkinter as tk
from gz.transport13 import Node
from gz.msgs10.double_v_pb2 import Double_V
from gz.msgs10.stringmsg_pb2 import StringMsg
from usv_vla.core import limited_action


def main():
    node = Node()
    publisher = node.advertise("/usv/control_request", Double_V)
    root = tk.Tk()
    root.title("无人艇示范采集：手动推力")
    throttle, steering = tk.DoubleVar(value=0), tk.DoubleVar(value=0)
    status = tk.StringVar()
    task_status = tk.StringVar(value="等待仿真传感器及任务初始化……")
    task_lock = threading.Lock()
    latest_task = {}
    def receive(msg):
        try:
            value=json.loads(msg.data)
            with task_lock:
                latest_task.clear()
                latest_task.update(value)
        except (ValueError,TypeError):
            pass
    node.subscribe(StringMsg,"/usv/task_status",receive)
    tk.Label(root,text="油门 N：W/S 调整；转向 N：A/D 调整；空格推力归零").pack(padx=15,pady=10)
    for name, variable, limit in [("油门",throttle,15),("转向（正值左转）",steering,7.5)]:
        tk.Scale(root,label=name,variable=variable,from_=-limit,to=limit,resolution=.5,
                 orient="horizontal",length=560).pack()
    tk.Label(root,textvariable=status).pack(pady=10)
    tk.Label(root,textvariable=task_status,justify="left",wraplength=560).pack(padx=15,pady=10)
    def send():
        action = limited_action([throttle.get()-steering.get(),throttle.get()+steering.get()])
        publisher.publish(Double_V(data=action))
        return action
    def stop(*_):
        throttle.set(0)
        steering.set(0)
        send()
    def close(*_):
        stop()
        root.destroy()
    def adjust(variable, delta, limit):
        variable.set(max(-limit,min(limit,variable.get()+delta)))
    for keys,variable,delta,limit in [(("w","<Up>"),throttle,.5,15),(("s","<Down>"),throttle,-.5,15),
                                      (("a","<Left>"),steering,.5,7.5),(("d","<Right>"),steering,-.5,7.5)]:
        for key in keys:
            root.bind(key,lambda event,v=variable,d=delta,l=limit:adjust(v,d,l))
    root.bind("<space>",stop)
    root.bind("<Escape>",close)
    root.protocol("WM_DELETE_WINDOW",close)
    signal.signal(signal.SIGINT,close)
    def tick():
        action=send()
        status.set(f"请求左 {action[0]:+.1f} N，右 {action[1]:+.1f} N；任务成功后记录器自动停止")
        with task_lock:
            value=dict(latest_task)
        if value:
            task_status.set(f"阶段：{value['phase']}\n目标距离 {value['distance_m']:.2f} m；"
                            f"艇速 {value['speed_mps']:.2f} m/s；低速保持 {value['hold_s']:.1f} s\n"
                            f"正确过门：{value['gate_passed']}；结束：{value['termination_reason']}\n"
                            f"{value['prompt']}")
        root.after(50,tick)
    tick()
    root.mainloop()


if __name__ == "__main__":
    main()
