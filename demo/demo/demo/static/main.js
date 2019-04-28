

var populate = (data)=>{
    let div = $(".result_cls");
    div.empty();
    var sample =  ` <a href="#" class="list-group-item list-group-item-action flex-column align-items-start">
    <div class="d-flex w-100 justify-content-between">
      <h5 class="mb-1"></h5>
      <small class="text-muted"></small>
    </div>
  </a>`;
    for (let i in data){
        let val = data[i];
        let child = $($.parseHTML(sample));
        child.find("h5").text(val.question);
        child.find("small").text(val.score);
        div.append(child);
    }

}

var onenter = ()=>{

    let data = $("#search_box").text();
    data = {"query":data};
    data = JSON.stringify(data);
    $.post("/matching/",data).done((resp)=>{
        resp=JSON.parse(resp);
        populate(resp);
    });

}
$(()=>{
    $(".search_icon").click(()=>{onenter();});
    $(document).keypress(function(event){
	
        var keycode = (event.keyCode ? event.keyCode : event.which);
        if(keycode == '13'){
            $(document).keypress(function(event){
	
                var keycode = (event.keyCode ? event.keyCode : event.which);
                if(keycode == '13'){
                    onenter();	
                }
                
            });
        }
        
    });
});