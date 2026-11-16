package main

import (
	"github.com/spf13/cobra"

	"github.com/shra26/ollama-compatibility-checker/internal/output"
)

func newCatalogCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "catalog",
		Short: "List models in the catalog",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			path, _ := cmd.Flags().GetString("catalog")
			ms, err := loadCatalog(path)
			if err != nil {
				return err
			}
			if asJSON, _ := cmd.Flags().GetBool("json"); asJSON {
				return output.WriteJSON(cmd.OutOrStdout(), ms)
			}
			output.RenderCatalog(cmd.OutOrStdout(), ms)
			return nil
		},
	}
}
