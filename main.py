import gradio as gr

from chat import run_agent_executor

iface = gr.ChatInterface(run_agent_executor)
iface.launch()

# could you create a folder on my desktop with the name "plane" and in the folder create 3 different files that contain each a different poem about plains. the name of the file should be the title of the poem.

