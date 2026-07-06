use renderer::Renderer;

pub fn install_dogfood_event_logger(renderer: &mut Renderer) {
    renderer::install_app_event_logger(renderer, "dogfood");
}
