from nicegui import ui
ui.label("Hello world")
ui.run()

ui.icon("thumb_up")
ui.markdown('This is **Markdown**')
ui.html("This is <strong> HTML </strong>")
with ui.row() :
    ui.label('css').style('color : #888; font-weight: bold')
    ui.label('tailwind').classes('font-serif')
    ui.label('Quasar').classes('q-ml-xl')
ui.link('Nice GUI on github', 'https://nicegui.io/#examples')
ui.run()