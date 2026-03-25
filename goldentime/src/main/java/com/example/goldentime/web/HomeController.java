package com.example.goldentime.web;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class HomeController {

    @GetMapping(value = "{path:[^\\.]*}")
    public String forward() {
        return "forward:/index.html";
    }

    @GetMapping(value = "/{path1}/{path2:[^\\.]*}")
    public String forwardDepth2() {
        return "forward:/index.html";
    }
}
