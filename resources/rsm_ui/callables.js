//cpp to js calls

//$('#mirror_x').prop('checked', mirror_x);
//$('#intensity_slider').slider('value', intensity * 100);

function bindings_complete()
{
  init_js();
}

function cpp_to_js(str)
{

}

function set_player_health( x )
{
  $('#health').html(x);

  if( x < 0 )
  {
    $('#health_text').hide();
    $('#game_over').show();
  }
}
