package com.example.goldentime.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class CorsConfig implements WebMvcConfigurer {

    @Value("${app.cors.allowed-origins:}")
    private String allowedOrigins;

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        if (allowedOrigins == null || allowedOrigins.isBlank()) {
            return;
        }

        String[] origins = allowedOrigins.split("\\s*,\\s*");

        // API
        registry.addMapping("/api/**")
                .allowedOrigins(origins)
                .allowedMethods("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS")
                .allowedHeaders("*")
                .allowCredentials(true);

        // 업로드/정적 파일 (프론트 dev 서버(5173 등)에서 직접 접근 시 CORS 필요)
        registry.addMapping("/images/**")
                .allowedOrigins(origins)
                .allowedMethods("GET", "OPTIONS")
                .allowedHeaders("*")
                .allowCredentials(true);

        registry.addMapping("/videos/**")
                .allowedOrigins(origins)
                .allowedMethods("GET", "OPTIONS")
                .allowedHeaders("*")
                .allowCredentials(true);
    }
}
