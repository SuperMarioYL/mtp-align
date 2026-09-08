from mtp_align.scheduler import MTPScheduler,ToolCall,Token,FlushDecision
scheduler=MTPScheduler(window_size=4,max_batch=8)
scheduler.queue_tool_call(ToolCall(id=1,name='lookup',args={'key':'demo'}))
for offset in range(4):
 decision=scheduler.on_token(Token(offset=offset,content=str(offset)))
 print(f'token {offset}: {decision.value}')
 if decision is FlushDecision.FLUSH:print('flushed calls:',[c.name for c in scheduler.flush()])
print('windows completed:',scheduler.windows_completed())
